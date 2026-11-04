"""Performance-gated progression within one run. No checkpoint/teacher dependency."""
class Curriculum:
    def __init__(self, cfg):
        self.cfg = cfg
        self.level = 0
        self.streak = 0
        self.last_probe_update = -1
        self.probes = []

    def probe_interval(self):
        return (self.cfg.bootstrap_probe_interval or self.cfg.probe_interval) if self.level < 2 else self.cfg.probe_interval

    def probe_due(self, updates):
        return updates > 0 and updates % self.probe_interval() == 0 and updates > self.last_probe_update

    def next_probe_update(self, updates):
        return (updates//self.probe_interval()+1)*self.probe_interval()

    def observe_probe(self, updates, successes, count, level):
        if count < self.cfg.probe_count or not 0 <= successes <= count:
            raise ValueError('Insufficient or invalid deterministic probe')
        if level != self.level or updates <= self.last_probe_update:
            raise ValueError('Stale curriculum probe')
        self.last_probe_update = updates
        rate = successes/count
        self.streak = self.streak+1 if rate >= self.cfg.curriculum_pass_rate else 0
        record = dict(updates=updates, level=level, success=successes, count=count,
                      rate=rate, kind='development_curriculum_probe_not_independent_validation')
        eligible = self.streak >= self.cfg.curriculum_consecutive and self.level < self.cfg.max_course_level
        record['promotion_deferred_at_budget'] = bool(eligible and updates >= self.cfg.iterations)
        if eligible and updates < self.cfg.iterations:
            self.level += 1
            self.streak = 0
        record['next_level'] = self.level
        self.probes.append(record)
        return record

    def state_dict(self):
        return dict(level=self.level, streak=self.streak, last_probe_update=self.last_probe_update,
                    probes=list(self.probes))

    def load_state_dict(self, state):
        if state['level'] not in range(self.cfg.max_course_level+1) or state['streak'] < 0:
            raise ValueError('Invalid curriculum state')
        self.level, self.streak = state['level'], state['streak']
        self.last_probe_update = state['last_probe_update']
        self.probes = list(state['probes'])
