"""One-off regression: audit relative dir_fd names without actually deleting any file."""

import hashlib
import os
from pathlib import Path

import independent_failure_audit as audit


here = Path(__file__).resolve().parent
fixture = here / "dirfd_fixture.txt"
before = hashlib.sha256(fixture.read_bytes()).hexdigest()
descriptor = os.open(here, os.O_RDONLY)
try:
    audit.audit_io("os.remove", (fixture.name, descriptor))
finally:
    os.close(descriptor)
relative = fixture.relative_to(audit.ROOT).as_posix()
audit.audit_io("os.remove", (relative, None))
audit.audit_io("os.remove", (relative, -1))
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == before
assert all(value == 0 for value in audit.COUNTS.values())
print("DIR_FD_RELATIVE_AND_NONE_NEGATIVE_ONE_TRUE_PATH_NO_DELETE 3")
