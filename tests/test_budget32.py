"""32帧边界与冻结两遍预算的无模型回归。"""
import unittest
from unittest.mock import patch
import numpy as np
from videoqa_budget32.frames import padded_count,validate_frames,native_frames
from videoqa_budget32.selection import two_pass,configure_visual_precision,reusable_rgb_hashes


class Budget32Tests(unittest.TestCase):
    def setUp(self):
        self.video=dict(start_pts=0,time_base='1/10',duration_fraction='10',duration_seconds=10.)

    def test_internal_padding(self):
        for n in (1,11,14,16,31,32):self.assertEqual(padded_count(n),n+n%2)
        for n in (0,33):
            with self.assertRaises(ValueError):padded_count(n)

    def test_native32_not_truncated(self):
        frames=native_frames(np.arange(100),self.video)
        self.assertEqual(len(frames),32)
        self.assertEqual([f['source_frame_index'] for f in frames],np.linspace(0,99,32,dtype=int).tolist())
        validate_frames(dict(video=self.video,selected_frames=frames))

    def test_short_source_unique(self):
        frames=native_frames(np.arange(11),self.video)
        self.assertEqual(len(frames),11)

    def test_vfr_exact_pts(self):
        pts=np.array([0,1,3,8,12,19,23,31,39,60,98])
        frames=native_frames(pts,self.video)
        self.assertEqual(frames[-1]['timestamp_seconds'],9.8)
        frames[-1]['source_pts']=frames[-2]['source_pts']
        with self.assertRaises(AssertionError):validate_frames(dict(video=self.video,selected_frames=frames))

    def test_two_pass_quota_is_forwarded(self):
        pool=dict(candidates=[],plan={},video={});quota={'0':32}
        with patch('videoqa_budget32.selection.select_density_decay',return_value=([],{'region_budgets':quota})) as first:
            with patch('videoqa_budget32.selection.select_local_distance',return_value=([1],{})) as second:
                result=two_pass(pool,np.empty((0,768)),{'frame_budget':32})
                self.assertEqual(result[0],[1]);self.assertEqual(second.call_args.args[-1],quota)
                self.assertEqual(first.call_args.args[-1]['frame_budget'],32)

    def test_visual_precision_does_not_depend_on_defaults(self):
        from types import SimpleNamespace as NS
        seeds=[];torch=NS(manual_seed=seeds.append,backends=NS(cuda=NS(matmul=NS(allow_tf32=True)),cudnn=NS(allow_tf32=True)))
        result=configure_visual_precision(torch)
        self.assertEqual(seeds,[2027]);self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32);self.assertEqual(result['feature_dtype'],'float32')

    def test_rgb_recovery_uses_source_identity_not_candidate_id(self):
        a=dict(candidate_index=0,source_frame_index=5,source_pts=50)
        b=dict(candidate_index=90,source_frame_index=5,source_pts=50)
        export=dict(selection=dict(selected_frames=[a]),rgb_sha256=['hash-a'])
        self.assertEqual(reusable_rgb_hashes([export],[b]),['hash-a'])
        self.assertIsNone(reusable_rgb_hashes([export],[dict(b,source_pts=51)]))
        bad=dict(selection=dict(selected_frames=[a]),rgb_sha256=['hash-b'])
        with self.assertRaises(AssertionError):reusable_rgb_hashes([export,bad],[a])
