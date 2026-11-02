import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_experiment import ROOT, load_config, prepare, main


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def args(self, dataset='THUMOS', feature='resnet50', variant='sacg', action='check'):
        rel = 'configs/{}/{}_{}.yaml'.format(dataset, variant, feature)
        c = load_config(ROOT / rel)
        names = ['class_' + str(i) for i in range(31 if dataset == 'TVSeries' else 22 if dataset == 'THUMOS' else 106)]
        info = {dataset: {'data_root': str(self.base), 'class_names': names, 'num_classes': len(names), 'ignore_index': 21 if dataset == 'THUMOS' else -100, 'metrics': 'cAP' if dataset == 'TVSeries' else 'AP', 'fps': 1 if dataset == 'CrossTask' else 4, 'train_session_set': ['synthetic_train'], 'test_session_set': ['synthetic_test']}}
        meta = self.base / 'metadata.json';meta.write_text(json.dumps(info))
        for d in [c.INPUT.VISUAL_FEATURE, c.INPUT.MOTION_FEATURE, c.INPUT.TARGET_PERFRAME]:
            folder = self.base / d;folder.mkdir(exist_ok=True)
            # Path-check fixtures only. These intentionally are NOT real arrays.
            for sid in ['synthetic_train','synthetic_test']:(folder/(sid+'.npy')).touch()
        return argparse.Namespace(config=rel,data_info=str(meta),data_root=str(self.base),output=str(self.base/'outputs with spaces'),seed=0,session=None,action=action,checkpoint=None,workers=None,gpu='0')

    def test_all_eight_configs_resolve_with_original_parser(self):
        for rel, entry in json.loads((ROOT/'configs/INDEX.json').read_text()).items():
            with self.subTest(rel=rel):
                a=self.args(entry['dataset'],entry['feature'],entry['variant'])
                c, cmd=prepare(a)
                self.assertGreater(c.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES,0)
                self.assertEqual(c.INPUT.MODALITY,'visual+motion')
                self.assertEqual(cmd[cmd.index('--config_file')+1],rel)
                self.assertEqual(c.MODEL.LSTR.INFERENCE_MODE,'batch')

    def test_baseline_sacg_pair_only_changes_generator_treatment(self):
        index=json.loads((ROOT/'configs/INDEX.json').read_text())
        for rel,e in index.items():
            if e['variant']!='sacg':continue
            a=load_config(ROOT/rel)
            b=load_config(ROOT/rel.replace('/sacg_', '/baseline_'))
            self.assertEqual(a.MODEL.LSTR.GEN_CROSS_ATTN_TYPE,'assa')
            self.assertEqual(a.MODEL.LSTR.GEN_CROSS_ASSA_INIT_W,[0.0,0.0])
            self.assertTrue(a.MODEL.LSTR.GEN_CONTRANORM)
            self.assertEqual(a.MODEL.LSTR.GEN_CONTRANORM_POS,'cross')
            self.assertEqual(a.MODEL.LSTR.GEN_CONTRANORM_SCALE,e['cfn_scale'])
            self.assertEqual(b.MODEL.LSTR.GEN_CONTRANORM_POS,'none')
            for key in ['GEN_CROSS_ATTN_TYPE','GEN_CONTRANORM','GEN_CONTRANORM_POS','GEN_CONTRANORM_SCALE']:
                a.MODEL.LSTR[key]=b.MODEL.LSTR[key]
            self.assertEqual(a.dump(),b.dump())

    def test_missing_real_metadata_stops_before_output_creation(self):
        a=self.args();Path(a.data_info).unlink()
        with self.assertRaisesRegex(ValueError,'Missing real data_info'):prepare(a)
        self.assertFalse(Path(a.output).exists())

    def test_overlap_rejected(self):
        a=self.args();p=Path(a.data_info);d=json.loads(p.read_text());d['THUMOS']['test_session_set']=['synthetic_train'];p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'overlap'):prepare(a)

    def test_missing_target_not_hidden_by_present_features(self):
        a=self.args();(self.base/'target_perframe/synthetic_test.npy').unlink()
        with self.assertRaisesRegex(ValueError,'Missing .npy'):prepare(a)

    def test_eval_requires_checkpoint_and_handles_spaces(self):
        a=self.args(action='eval')
        with self.assertRaisesRegex(ValueError,'requires --checkpoint'):prepare(a)
        p=self.base/'checkpoint with spaces.pth';p.touch();a.checkpoint=str(p)
        c,cmd=prepare(a)
        self.assertEqual(c.TEST,1)
        self.assertEqual(cmd[cmd.index('MODEL.CHECKPOINT')+1],str(p.resolve()))
        self.assertTrue(c.OUTPUT_DIR.endswith('eval_seed0'))

    def test_tvseries_uses_drop_oob_and_cap(self):
        a=self.args('TVSeries','drop_oob');c,_=prepare(a)
        self.assertEqual(c.INPUT.TARGET_PERFRAME,'target_perframe_drop_oob')
        self.assertEqual(c.DATA.METRICS,'cAP')
        self.assertEqual(c.DATA.NUM_CLASSES,31)

    def test_existing_run_is_not_reused(self):
        a=self.args(action='train');c,_=prepare(a);dest=Path(c.OUTPUT_DIR);dest.mkdir(parents=True)
        sentinel=dest/'preserve.txt';sentinel.write_text('original')
        with patch('scripts.run_experiment.subprocess.run') as run,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            code=main(['train','--config',a.config,'--data-info',a.data_info,'--data-root',a.data_root,'--output',a.output])
        self.assertEqual(code,2)
        self.assertEqual(sentinel.read_text(),'original')
        self.assertEqual(run.call_count,1) # dependency probe only, no training launched

    def test_copied_source_hashes_match_manifest(self):
        for record in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT/record['release_relative_path']).read_bytes()).hexdigest(),record['sha256'])


if __name__=='__main__':
    unittest.main()
