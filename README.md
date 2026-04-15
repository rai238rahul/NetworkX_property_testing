# NetworkX property testing

E0 251o-style property-based tests for NetworkX (Hypothesis + `pytest`).

## Dependencies

Install [NetworkX](https://networkx.org/), [Hypothesis](https://hypothesis.readthedocs.io/), and [pytest](https://pytest.org/), for example:

```bash
pip install networkx hypothesis pytest
```

## Run tests

From the repository root:

```bash
python -m pytest test_mst_properties.py -v
```
