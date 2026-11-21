"""CrowdShield entrypoint.

For A1, this only proves config + logging + seeding wire together.
No video, no models — that starts in Phase B.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config.loader import load_config
from logging_setup import setup_logging
from seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield AI/ML pipeline")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--venue", default=None, help="Path to venue override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    config_path = Path(args.config) if args.config else None
    venue_path = Path(args.venue) if args.venue else None

    # Single load path: no arguments -> default; --config -> that config;
    # --venue -> default + venue override; both -> config + venue override.
    cfg = load_config(config_path, venue_path)

    set_global_seed(cfg.random_seed)

    logger.info("CrowdShield config loaded (version=%s)", cfg.config_version)
    logger.info("Venue: %s | Zones configured: %d", cfg.venue.name, len(cfg.venue.zones))
    logger.info("Detection model: %s | device=%s", cfg.detection.model, cfg.detection.device)
    logger.info("Startup complete — no models loaded yet (Phase A only).")


if __name__ == "__main__":
    main()
