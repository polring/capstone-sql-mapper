import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import asdict, fields, replace
from pathlib import Path
from unittest.mock import patch

from src.back.config import ModelConfig
from training.back.config import TrainConfig
from training.back.train import parse_args, read_configs, training_split_paths


class TrainingConfigCliTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "run.json"
        self.cli = {"config": str(self.path), "data_dir": "data/v1", "run_dir": "runs/example",
                    "seed": "19", "device": "cpu"}

    @staticmethod
    def argv(options):
        return [token for name, value in options.items() for token in ("--" + name.replace("_", "-"), value)]

    def args_for(self, settings, **cli):
        self.path.write_text(json.dumps(settings), encoding="utf-8")
        return parse_args(self.argv(self.cli | cli))

    def test_normal_and_memorization_defaults(self):
        args = self.args_for({})
        self.assertEqual(args.data_dir, Path("data/v1"))
        self.assertEqual(args.run_dir, Path("runs/example"))
        self.assertIsNone(args.overfit)
        self.assertEqual(read_configs(args, None), (ModelConfig(), TrainConfig(seed=19)))
        args = self.args_for({"overfit": 32})
        self.assertEqual(args.overfit, 32)
        self.assertEqual(read_configs(args, None), (ModelConfig(), TrainConfig(seed=19, epochs=400, patience=0)))

    def test_paths_seed_and_device_required_on_cli_including_resume(self):
        self.path.write_text("{}", encoding="utf-8")
        for mode in ({}, {"resume": "runs/example/last.pt"}):
            for key in self.cli:
                options = self.cli | mode
                del options[key]
                stderr = io.StringIO()
                with self.subTest(mode=mode, missing=key), redirect_stderr(stderr), self.assertRaises(SystemExit):
                    parse_args(self.argv(options))
                self.assertIn("--" + key.replace("_", "-"), stderr.getvalue().split("error: ")[1])

    def test_directory_selects_one_complete_pair_or_an_explicit_training_file(self):
        directory = self.path.parent
        args = self.args_for({}, data_dir=str(directory))
        train = directory / "v1.[test]_back_train.json"
        validation = directory / "v1.[test]_back_validation.json"
        train.touch()
        validation.touch()
        legacy_train = directory / "back_train.json"
        legacy_validation = directory / "back_validation.json"
        legacy_train.touch()
        self.assertEqual(training_split_paths(args), (train, validation))
        legacy_validation.touch()
        with self.assertRaisesRegex(ValueError, "multiple datasets.*--train-file"):
            training_split_paths(args)

        options = self.cli | {"train_file": str(train)}
        del options["data_dir"]
        self.assertEqual(training_split_paths(parse_args(self.argv(options))), (train, validation))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(self.argv(options | {"data_dir": str(directory)}))

        train.unlink()
        self.assertEqual(training_split_paths(args), (legacy_train, legacy_validation))

    def test_overfit_uses_training_file_without_validation(self):
        directory = self.path.parent
        train = directory / "v1_back_train.json"
        train.touch()
        args = self.args_for({}, data_dir=str(directory))
        with self.assertRaisesRegex(ValueError, "no matching dataset splits"):
            training_split_paths(args)
        args = self.args_for({"overfit": 4}, data_dir=str(directory))
        self.assertEqual(training_split_paths(args)[0], train)

    def test_cli_settings_rejected_in_json(self):
        for key in ("data_dir", "train_file", "run_dir", "resume", "device", "seed"):
            with self.subTest(key=key), redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
                self.args_for({key: "old value"})
            self.assertIn("command line: --" + key.replace("_", "-"), stderr.getvalue())
        with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            self.args_for({"train_config": {"seed": 19}})
        self.assertIn("command line with --seed", stderr.getvalue())

    def test_run_settings_and_independent_model_and_optimizer_values(self):
        model = ModelConfig(vocab_size=768, d_model=96, num_layers=2, num_heads=4,
                            ffn_dim=192, context_length=256, dropout=0.1, eps=1e-5, theta=8000)
        train = TrainConfig(batch_size=4, accumulation_steps=2, epochs=3, lr=0.002, min_lr=0.0002,
                            warmup_ratio=0.05, weight_decay=0.2, beta1=0.85, beta2=0.98, eps=1e-9,
                            max_grad_norm=0.9, patience=1, seed=17, precision="fp32")
        train_settings = asdict(train)
        del train_settings["seed"]
        args = self.args_for({
            "threads": 3,
            "overfit": 8, "eval_every": 2, "eval_batch_size": 4,
            "model_config": asdict(model), "train_config": train_settings,
        }, data_dir="data/v3", seed="17")
        self.assertEqual(Path(args.data_dir), Path("data/v3"))
        self.assertEqual(Path(args.run_dir), Path("runs/example"))
        self.assertEqual((args.device, args.threads, args.overfit, args.eval_every, args.eval_batch_size),
                         ("cpu", 3, 8, 2, 4))
        self.assertEqual(read_configs(args, None), (model, train))

    def test_resume_uses_saved_settings_and_can_extend_epochs(self):
        model = ModelConfig(num_layers=3)
        train = TrainConfig(lr=4e-4, seed=19)
        saved = {"model_config": asdict(model), "train_config": asdict(train)}
        path = "runs/example/last.pt"
        args = self.args_for({"train_config": {"epochs": 15}}, resume=path)
        self.assertEqual(Path(args.resume), Path(path))
        self.assertEqual(read_configs(args, saved), (model, replace(train, epochs=15)))
        args = self.args_for({"model_config": {"num_layers": 8}}, resume=path)
        with self.assertRaises(ValueError):
            read_configs(args, saved)
        args = self.args_for({}, resume=path, seed="20")
        with self.assertRaisesRegex(ValueError, "--seed must match the checkpoint seed"):
            read_configs(args, saved)

    def test_device_selection_supports_cpu_gpu_and_auto(self):
        for requested, expected in (("cpu", "cpu"), ("gpu", "cuda"), ("cuda", "cuda"), ("cuda:0", "cuda:0")):
            with self.subTest(device=requested):
                self.assertEqual(self.args_for({}, device=requested).device, expected)
        for cuda_available in (True, False):
            with patch("torch.cuda.is_available", return_value=cuda_available):
                args = self.args_for({}, device="auto")
                self.assertEqual(args.device, "cuda" if cuda_available else "cpu")

    def test_top_level_description_does_not_change_training_options(self):
        settings = {"train_config": {"epochs": 3}}
        self.assertEqual(self.args_for(settings | {"description": "학습 설정 설명"}), self.args_for(settings))

    def test_null_settings_match_omitted_options(self):
        settings = {
            "overfit": None, "threads": None, "eval_every": None, "eval_batch_size": None,
            "model_config": {field.name: None for field in fields(ModelConfig)},
            "train_config": {field.name: None for field in fields(TrainConfig) if field.name != "seed"},
        }
        self.assertEqual(self.args_for(settings), self.args_for({}))
        self.assertEqual(self.args_for(settings | {"overfit": 4}), self.args_for({"overfit": 4}))
        settings["train_config"]["epochs"] = 7
        self.assertEqual(self.args_for(settings), self.args_for({"train_config": {"epochs": 7}}))

    def test_resume_reuses_original_json_after_context_expansion(self):
        model = ModelConfig(context_length=600)
        saved = {"model_config": asdict(model), "train_config": asdict(TrainConfig(seed=19)),
                 "requested_context_length": 128}
        checkpoint = "runs/example/last.pt"
        args = self.args_for({"model_config": {"context_length": 128}}, resume=checkpoint)
        self.assertEqual(read_configs(args, saved)[0], model)
        args = self.args_for({"model_config": {"context_length": 256}}, resume=checkpoint)
        with self.assertRaises(ValueError):
            read_configs(args, saved)

    def test_hyperparameters_are_json_only(self):
        for key in ("overfit", "epochs"):
            with self.subTest(key=key), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args_for({}, **{key: "2"})

    def test_invalid_cli_values(self):
        for overrides in ({"data_dir": ""}, {"run_dir": " "}, {"resume": " "}, {"device": "invalid"}, {"seed": "abc"}):
            with self.subTest(overrides=overrides), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args_for({}, **overrides)

    def test_rejects_unknown_keys_and_invalid_data_types(self):
        for settings in (
            [], {"lr": 0.01}, {"threadz": 2}, {"model_config": {"layerz": 4}},
            {"extra": None}, {"device": None}, {"model_config": {"layers": None}}, {"train_config": {"seed": None}},
            {"description": None}, {"description": 42}, {"train_config": {"description": "설명"}},
            {"train_config": {"batch-size": 4}}, {"train_config": "cpu"},
            {"train_config": {"batch_size": True}}, {"overfit": "32"}, {"threads": False},
            {"model_config": {"d_model": 256.5}}, {"train_config": {"precision": "fast"}},
            {"train_config": {"lr": float("inf")}}, {"model_config": {"eps": float("nan")}},
        ):
            with self.subTest(settings=settings), redirect_stderr(io.StringIO()), self.assertRaises((SystemExit, ValueError)):
                read_configs(self.args_for(settings), None)

    def test_rejects_wrong_extension_and_malformed_json(self):
        for name, contents in (("train.txt", "{}"), ("broken.json", "{,}"),
                               ("duplicate.json", '{"train_config":{"lr":0.001,"lr":0.002}}')):
            path = self.path.with_name(name)
            path.write_text(contents, encoding="utf-8")
            with self.subTest(name=name), redirect_stderr(io.StringIO()), self.assertRaises((SystemExit, ValueError)):
                parse_args(self.argv(self.cli | {"config": str(path)}))


if __name__ == "__main__":
    unittest.main()
