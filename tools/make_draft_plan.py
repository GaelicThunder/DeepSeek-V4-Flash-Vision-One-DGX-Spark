#!/usr/bin/env python3
"""Synthesize a REAP-plan-shaped JSON for build_dspark_draft.py on an UNPRUNED (256-expert) checkpoint.
Ranking = the 64 original expert ids of the current 0731 K64 draft (DSPARK_DRAFT_PLAN.json), same order.
Everything else (192 experts) follows in ascending id so any --experts up to 255 works."""
import json, sys
src_plan, out = sys.argv[1], sys.argv[2]
sel = json.load(open(src_plan))["selected_original_expert_ids"]
assert len(sel) == len(set(sel)) == 64 and max(sel) < 256
rest = [e for e in range(256) if e not in set(sel)]
ranking = sel + rest
plan = {
  "note": "synthetic plan for dsvision: identity keep (256 experts), ranking seeded with the 0731 K64 draft selection",
  "keep_maps": {
    "mtp_keep": list(range(256)),
    "mtp_keep_from_layer": "42",
    "mtp_ranked": ranking,
    "structured_ranked_by_category": {
      "agentic_tool_trajectory": {"42": ranking},
      "tool_calling": {"42": ranking},
    },
  },
}
json.dump(plan, open(out, "w"), indent=1)
print("plan written:", out, "first 8:", ranking[:8])
