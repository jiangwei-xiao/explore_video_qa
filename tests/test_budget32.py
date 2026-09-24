"""32帧边界与冻结两遍预算的无模型回归。"""
import unittest
from unittest.mock import patch
import numpy as np
from videoqa_budget32.frames import padded_count,validate_frames,native_frames
from videoqa_budget32.selection import two_pass


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
