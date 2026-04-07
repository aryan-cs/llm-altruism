"""Tests for society simulation mechanics."""

from src.simulation import AgentState, EconomyEngine, World, WorldConfig


def build_world() -> World:
    """Create a tiny world with two live agents for unit tests."""
    world = World(WorldConfig(initial_agent_resources=5, initial_public_resources=10))
    world.add_agent(AgentState(agent_id="agent-a", model="test", resources=5))
    world.add_agent(AgentState(agent_id="agent-b", model="test", resources=5))
    return world


def test_world_transfer_to_self_is_noop():
    """Self-transfers should not move resources."""
    world = build_world()

    moved = world.transfer("agent-a", "agent-a", 3)

    assert moved == 0
    assert world.get_state("agent-a").resources == 5


def test_economy_share_to_self_is_ignored():
    """Self-share actions should not create fake social events or trade volume."""
    world = build_world()
    economy = EconomyEngine()

    events, trade_volume = economy.resolve(
        world,
        {"agent-a": {"action": "share", "target": "agent-a", "amount": 3}},
        allow_private_messages=True,
        allow_steal=True,
    )

    assert trade_volume == 0
    assert events == []
    assert world.get_state("agent-a").resources == 5


def test_economy_steal_from_self_is_ignored():
    """Self-steal actions should be ignored entirely."""
    world = build_world()
    economy = EconomyEngine()

    events, trade_volume = economy.resolve(
        world,
        {"agent-a": {"action": "steal", "target": "agent-a", "amount": 2}},
        allow_private_messages=True,
        allow_steal=True,
    )

    assert trade_volume == 0
    assert events == []
    assert world.get_state("agent-a").resources == 5
