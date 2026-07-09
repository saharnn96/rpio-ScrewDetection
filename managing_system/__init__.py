"""MANAGING SYSTEM - the MAPLE-K self-adaptation loop.

    nodes.py      Monitor / Analysis / Plan / Legitimate / Execute (rpclpy
                  nodes; require a redis server on 127.0.0.1:6379)
    messages.py   the K: knowledge classes flowing through the store
    config.yaml   node wiring (events/knowledge/redis) + Adaptation_Config
                  (entropy threshold, window, candidate model, replan budget)

This package adapts the managed system (`managed_system/`) exclusively through
the probe/effector methods of ScrewDetectionCore, injected via
`managed_system.core.set_core()` at the composition root. It knows nothing
about cameras, robots or YOLO.
"""
