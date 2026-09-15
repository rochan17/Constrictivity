"""
test_throat_removal.py
=======================

Unit tests for the network-mutation logic in throat_removal.py, using a
small hand-built PoreNetwork rather than a full microstructure (fast, no
data/ dependency). The prune_and_recompute end-to-end path is exercised
manually against a real microstructure (see the module docstring in
throat_removal.py) rather than here, since it requires a full solve.
"""

from pathlib import Path
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import particulate_claude as pc
from throat_removal import remove_throats


def _make_body(body_id, connected_bodies, connected_throats):
    return pc.PoreBody(
        body_id=body_id,
        nodal_state=0,
        volume_voxels=100,
        equivalent_radius=3.0,
        max_radius=3.0,
        centroid=(0.0, 0.0, 0.0),
        volume_centroid=(0.0, 0.0, 0.0),
        connected_bodies=list(connected_bodies),
        connected_throats=list(connected_throats),
    )


def _make_throat(throat_id, b1, b2):
    return pc.PoreThroat(
        throat_id=throat_id,
        connects=(b1, b2),
        volume_voxels=10,
    )


@pytest.fixture
def chain_network():
    """1 -(t1)- 2 -(t2)- 3, plus a lone extra throat t3: 2 -(t3)- 4.
    Body 4 has only throat t3, so removing t3 isolates it.
    """
    bodies = {
        1: _make_body(1, [2], [1]),
        2: _make_body(2, [1, 3, 4], [1, 2, 3]),
        3: _make_body(3, [2], [2]),
        4: _make_body(4, [2], [3]),
    }
    throats = {
        1: _make_throat(1, 1, 2),
        2: _make_throat(2, 2, 3),
        3: _make_throat(3, 2, 4),
    }
    return pc.PoreNetwork(bodies=bodies, throats=throats)


def test_remove_throats_updates_both_endpoints(chain_network):
    pruned = remove_throats(chain_network, [2])

    assert 2 not in pruned.throats
    assert 3 not in pruned.bodies[2].connected_bodies
    assert 2 not in pruned.bodies[3].connected_bodies
    assert 2 not in pruned.bodies[2].connected_throats
    assert 2 not in pruned.bodies[3].connected_throats


def test_remove_throats_leaves_other_bodies_untouched(chain_network):
    pruned = remove_throats(chain_network, [2])

    assert pruned.bodies[1].connected_bodies == [2]
    assert pruned.bodies[1].connected_throats == [1]
    assert set(pruned.bodies[4].connected_bodies) == {2}


def test_remove_throats_does_not_mutate_original(chain_network):
    remove_throats(chain_network, [2])

    assert 2 in chain_network.throats
    assert 3 in chain_network.bodies[2].connected_bodies


def test_remove_throats_isolates_body_but_keeps_it(chain_network):
    pruned = remove_throats(chain_network, [3])

    assert 4 in pruned.bodies
    assert pruned.bodies[4].coordination_number == 0
    assert pruned.bodies[4].num_throats == 0
    assert pruned.num_throats == 2


def test_remove_multiple_throats(chain_network):
    pruned = remove_throats(chain_network, [1, 2, 3])

    assert pruned.num_throats == 0
    for body in pruned.bodies.values():
        assert body.coordination_number == 0
        assert body.connected_bodies == []
        assert body.connected_throats == []


def test_remove_unknown_throat_id_is_a_noop(chain_network):
    pruned = remove_throats(chain_network, [999])

    assert pruned.num_throats == chain_network.num_throats
    assert pruned.bodies[2].connected_bodies == chain_network.bodies[2].connected_bodies
