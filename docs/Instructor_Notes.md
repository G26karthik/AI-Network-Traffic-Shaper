# Instructor Notes: AI Traffic Shaper (Do not distribute with student PBL)

- Good vs. bad:
  - Good: Strong, practical integration of networking and ML; clear, hands-on learning of capture, features, and inference; Windows-focused guidance is pragmatic.
  - Risks: Using dst_port as label risks leakage; synthetic traffic may not generalize; Wi‑Fi capture on Windows can be finicky and frustrate students—use loopback fallback.
- Difficulty: Intermediate. Networking setup plus ML pipeline is manageable but requires troubleshooting; shaping adds admin hurdles.
- Uniqueness: Moderate. Many labs do traffic classification, but combining live generation, capture auto-stop, Windows firewall shaping, and a clean Pipeline is cohesive.
- Real impact: Educationally high; production impact limited unless expanded to flow-level features, realistic datasets, and safer shaping (QoS tagging or rate-limiting) rather than blocking.

Suggested variations:
- Add flow statistics and time-based features for more realistic generalization.
- Assign an ablation study: remove dst_port to probe leakage; compare feature sets.
- Replace blocking with DSCP tagging simulations.
- Provide pre-captured pcap(s) for offline practice when capture drivers are unreliable.
