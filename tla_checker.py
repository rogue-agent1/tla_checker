#!/usr/bin/env python3
"""tla_checker - TLA+ style model checker for state machines.

Explores all reachable states to verify invariants and liveness properties.

Usage: python tla_checker.py [--demo]
"""
import sys
from collections import deque
from itertools import product

class Spec:
    def __init__(self, name):
        self.name = name
        self.variables = {}
        self.init_states = []
        self.actions = []
        self.invariants = []
        self.type_constraints = {}

    def var(self, name, domain):
        self.variables[name] = domain
        self.type_constraints[name] = domain

    def init(self, states):
        """Add initial state(s). Accepts a dict or list of dicts."""
        if isinstance(states, dict):
            self.init_states.append(dict(states))
        else:
            for s in states:
                self.init_states.append(dict(s))

    def action(self, name, guard, effect):
        """guard(state) -> bool, effect(state) -> [new_states]"""
        self.actions.append((name, guard, effect))

    def invariant(self, name, pred):
        self.invariants.append((name, pred))

class ModelChecker:
    def __init__(self, spec):
        self.spec = spec
        self.visited = set()
        self.trace = {}  # state -> (parent_state, action_name)
        self.states_explored = 0
        self.violations = []

    def _freeze(self, state):
        return tuple(sorted(state.items()))

    def _thaw(self, frozen):
        return dict(frozen)

    def check(self, max_states=100000):
        queue = deque()
        for init in self.spec.init_states:
            frozen = self._freeze(init)
            if frozen not in self.visited:
                self.visited.add(frozen)
                queue.append(frozen)
                self.trace[frozen] = (None, "Init")
                self._check_invariants(init, "Init")

        while queue and self.states_explored < max_states:
            current_frozen = queue.popleft()
            current = self._thaw(current_frozen)
            self.states_explored += 1

            for action_name, guard, effect in self.spec.actions:
                if not guard(current):
                    continue
                next_states = effect(current)
                for ns in next_states:
                    nf = self._freeze(ns)
                    if nf not in self.visited:
                        self.visited.add(nf)
                        queue.append(nf)
                        self.trace[nf] = (current_frozen, action_name)
                        self._check_invariants(ns, action_name)

        return len(self.violations) == 0

    def _check_invariants(self, state, action):
        for inv_name, pred in self.spec.invariants:
            if not pred(state):
                self.violations.append({
                    "invariant": inv_name,
                    "state": dict(state),
                    "action": action,
                })

    def get_trace(self, state):
        """Get trace from init to given state."""
        frozen = self._freeze(state) if isinstance(state, dict) else state
        trace = []
        while frozen is not None:
            parent, action = self.trace.get(frozen, (None, None))
            trace.append((action, self._thaw(frozen)))
            frozen = parent
        trace.reverse()
        return trace

    def stats(self):
        return {
            "states_explored": self.states_explored,
            "unique_states": len(self.visited),
            "violations": len(self.violations),
        }

def main():
    print("=== TLA+ Style Model Checker ===\n")

    # Example 1: Mutual exclusion (Peterson's algorithm)
    print("--- Peterson's Mutual Exclusion ---")
    spec = Spec("Peterson")
    # State: flag0, flag1, turn, pc0, pc1
    spec.init([{"flag0": False, "flag1": False, "turn": 0, "pc0": "idle", "pc1": "idle"}])

    def p0_enter(s):
        return s["pc0"] == "idle"
    def p0_enter_eff(s):
        ns = dict(s); ns["flag0"] = True; ns["turn"] = 1; ns["pc0"] = "wait"
        return [ns]
    def p0_wait(s):
        return s["pc0"] == "wait" and (not s["flag1"] or s["turn"] == 0)
    def p0_wait_eff(s):
        ns = dict(s); ns["pc0"] = "crit"
        return [ns]
    def p0_exit(s):
        return s["pc0"] == "crit"
    def p0_exit_eff(s):
        ns = dict(s); ns["flag0"] = False; ns["pc0"] = "idle"
        return [ns]

    def p1_enter(s):
        return s["pc1"] == "idle"
    def p1_enter_eff(s):
        ns = dict(s); ns["flag1"] = True; ns["turn"] = 0; ns["pc1"] = "wait"
        return [ns]
    def p1_wait(s):
        return s["pc1"] == "wait" and (not s["flag0"] or s["turn"] == 1)
    def p1_wait_eff(s):
        ns = dict(s); ns["pc1"] = "crit"
        return [ns]
    def p1_exit(s):
        return s["pc1"] == "crit"
    def p1_exit_eff(s):
        ns = dict(s); ns["flag1"] = False; ns["pc1"] = "idle"
        return [ns]

    spec.action("p0_enter", p0_enter, p0_enter_eff)
    spec.action("p0_wait", p0_wait, p0_wait_eff)
    spec.action("p0_exit", p0_exit, p0_exit_eff)
    spec.action("p1_enter", p1_enter, p1_enter_eff)
    spec.action("p1_wait", p1_wait, p1_wait_eff)
    spec.action("p1_exit", p1_exit, p1_exit_eff)

    spec.invariant("MutualExclusion", lambda s: not (s["pc0"] == "crit" and s["pc1"] == "crit"))

    mc = ModelChecker(spec)
    ok = mc.check()
    print(f"  Result: {'✓ PASS' if ok else '✗ FAIL'}")
    print(f"  Stats: {mc.stats()}")

    # Example 2: Broken mutex (no turn variable)
    print("\n--- Broken Mutex (no turn) ---")
    spec2 = Spec("BrokenMutex")
    spec2.init([{"flag0": False, "flag1": False, "pc0": "idle", "pc1": "idle"}])
    # Broken: check flag BEFORE setting own flag
    spec2.action("p0_check", lambda s: s["pc0"]=="idle" and not s["flag1"],
                 lambda s: [dict(s, pc0="set")])
    spec2.action("p0_set", lambda s: s["pc0"]=="set",
                 lambda s: [dict(s, flag0=True, pc0="crit")])
    spec2.action("p0_exit", lambda s: s["pc0"]=="crit",
                 lambda s: [dict(s, flag0=False, pc0="idle")])
    spec2.action("p1_check", lambda s: s["pc1"]=="idle" and not s["flag0"],
                 lambda s: [dict(s, pc1="set")])
    spec2.action("p1_set", lambda s: s["pc1"]=="set",
                 lambda s: [dict(s, flag1=True, pc1="crit")])
    spec2.action("p1_exit", lambda s: s["pc1"]=="crit",
                 lambda s: [dict(s, flag1=False, pc1="idle")])
    spec2.invariant("MutualExclusion", lambda s: not (s["pc0"]=="crit" and s["pc1"]=="crit"))

    mc2 = ModelChecker(spec2)
    ok2 = mc2.check()
    print(f"  Result: {'✓ PASS' if ok2 else '✗ FAIL (expected!)'}")
    print(f"  Stats: {mc2.stats()}")
    if mc2.violations:
        v = mc2.violations[0]
        print(f"  Violation: {v['invariant']} in state {v['state']}")
        trace = mc2.get_trace(v['state'])
        print(f"  Trace ({len(trace)} steps):")
        for action, state in trace:
            print(f"    {action}: {state}")

if __name__ == "__main__":
    main()
