"""Round22 changes sampled boundary exposure, not control or evaluation semantics."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config,ROOT
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL,state

class BoundaryRound22(unittest.TestCase):
    def test_reviewed_config_is_round18_plus_boundary_fraction(self):
        old=load_config(ROOT/'configs/round18.json').to_dict();new=load_config(ROOT/'configs/round22.json').to_dict()
        differences={k:(old[k],new[k]) for k in old if old[k]!=new[k]}
        self.assertEqual(differences,{'round':(18,22),'boundary_command_fraction':(.25,.5)})

    def test_more_edges_preserve_drawn_categories_signs_and_continuous_targets(self):
        cfg=load_config(ROOT/'configs/round22.json')
        a=Runtime(8192,'cpu',.06,replace(cfg,boundary_command_fraction=.25),MODEL)
        b=Runtime(8192,'cpu',.06,cfg,MODEL)
        a.level[:]=b.level[:]=8
        old=a.training_command(torch.arange(8192));new=b.training_command(torch.arange(8192));limits=b.stage_ranges[8,:2]
        torch.testing.assert_close(old.sign(),new.sign(),rtol=0,atol=0)
        unchanged=(new.abs()!=limits)
        torch.testing.assert_close(old[unchanged],new[unchanged],rtol=0,atol=0)
        self.assertGreater(int((new.abs()==limits).sum()),1.8*int((old.abs()==limits).sum()))
        self.assertGreater(int(((new.abs()>0)&(new.abs()<limits)).sum()),4000)
        torch.testing.assert_close(a.generator.get_state(),b.generator.get_state(),rtol=0,atol=0)

    def test_all_native_probe_levels_keep_original_requests_and_rng(self):
        cfg=load_config(ROOT/'configs/round22.json')
        for level in range(2,9):
            ps=[Runtime(32,'cpu',.06,replace(cfg,boundary_command_fraction=f),MODEL) for f in (.25,.5)]
            for p in ps:p.probe_level=level;p.probe_duration=60.;p.reset(torch.arange(32));p.control.phase[:]=Phase.BALANCE
            for _ in range(160):
                for p in ps:p.balance_elapsed+=.06;p.before_action(state(32))
                torch.testing.assert_close(ps[0].control.command,ps[1].control.command,rtol=0,atol=0)
                torch.testing.assert_close(ps[0].observation(state(32)),ps[1].observation(state(32)),rtol=0,atol=0)
            torch.testing.assert_close(ps[0].generator.get_state(),ps[1].generator.get_state(),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
