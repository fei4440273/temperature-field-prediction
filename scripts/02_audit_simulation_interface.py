#!/usr/bin/env python
import argparse
import json

from sic_cu.eval.interface_audit import audit_simulation_interface


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit same-coordinate SiC-Cu interface jumps")
    parser.add_argument("--output", default="reports/simulation_interface_audit.json")
    args = parser.parse_args()
    result = audit_simulation_interface(args.output)
    print(
        json.dumps(
            {
                "paired_nodes": result["paired_same_coordinate_nodes"],
                "global_max_abs_jump_c": result["global_max_abs_jump_c"],
                "global_max_record": result["global_max_record"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
