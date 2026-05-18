from asfuzz.spec.ops_catalog import make_matmul
from asfuzz.spec.validate import validate


def test_opspec_signature_stable():
    a = make_matmul(2, 3, 4)
    b = make_matmul(2, 3, 4)
    assert a.signature() == b.signature()
    assert validate(a).ok

