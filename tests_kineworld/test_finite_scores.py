import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('scores_test',Path(__file__).resolve().parents[1]/'evaluation/cosmos3/generator/paibench_g/run_motion_smoothness_sharded.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ScoreTests(unittest.TestCase):
    def test_nonfinite_and_boolean_rejected(self):
        for value in (float('nan'),float('inf'),float('-inf'),True):
            with self.assertRaises(ValueError):module.finite_score(value)
        self.assertEqual(module.finite_score(.8),.8)

    def test_resume_rejects_bad_score(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'worker_00.json').write_text(json.dumps([{'video_path':'a.mp4','video_results':float('nan')}]))
            with self.assertRaises(ValueError):module.collect_saved_records(p)

    def test_valid_merge_and_atomic_write_failure_preserves_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);paths=[str((p/n).absolute()) for n in ('a.mp4','b.mp4')]
            module.write_json_atomic(p/'worker_00.json',[{'video_path':v,'video_results':s} for v,s in zip(paths,[.6,.8])])
            result=p/'result.json'
            self.assertAlmostEqual(module.write_merged_result(output_dir=p,result_file=result,videos=paths),.7)
            before=result.read_bytes()
            with self.assertRaises(ValueError):module.write_json_atomic(result,{'bad':float('nan')})
            self.assertEqual(result.read_bytes(),before)

if __name__=='__main__':unittest.main()
