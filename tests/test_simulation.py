"""Tests for society simulation mechanics."""

from src.simulation import (
    AgentState,
    EconomyEngine,
    EventConfig,
    RandomEventEngine,
    World,
    WorldConfig,
)


def build_world() -> World:
    """Create a tiny world with two live agents for unit tests."""
    world = World(
        WorldConfig(
            initial_public_food=10,
            initial_public_water=10,
            initial_agent_food=5,
            initial_agent_water=5,
            initial_agent_energy=8,
            initial_agent_health=10,
        )
    )
    world.add_agent(AgentState(agent_id="agent-a", model="test", food=5, water=5, energy=8, health=10))
    world.add_agent(AgentState(agent_id="agent-b", model="test", food=5, water=5, energy=8, health=10))
    return world


def test_world_transfer_to_self_is_noop():
    """Self-transfers should not move resources."""
    world = build_world()

    moved = world.transfer("agent-a", "agent-a", "food", 3)

    assert moved == 0
    assert world.get_state("agent-a").food == 5


def test_economy_share_to_self_is_ignored():
    """Self-share actions should not create fake social events or trade volume."""
    world = build_world()
    economy = EconomyEngine()

    events, trade_volume = economy.resolve(
        world,
        {
            "agent-a": {
                "action": "share",
                "target": "agent-a",
                "resource_type": "food",
                "amount": 3,
            }
        },
        allow_private_messages=True,
        allow_steal=True,
    )

    assert trade_volume == 0
    assert events == []
    assert world.get_state("agent-a").food == 5


def test_economy_steal_from_self_is_ignored():
    """Self-steal actions should be ignored entirely."""
    world = build_world()
    economy = EconomyEngine()

    events, trade_volume = economy.resolve(
        world,
        {
            "agent-a": {
                "action": "steal",
                "target": "agent-a",
                "resource_type": "water",
                "amount": 2,
            }
        },
        allow_private_messages=True,
        allow_steal=True,
    )

    assert trade_volume == 0
    assert events == []
    assert world.get_state("agent-a").water == 5


def test_nightly_maintenance_consumes_food_water_and_recovers_sleep():
    """The nightly upkeep phase should consume essentials and reward sleep."""
    world = build_world()

    result = world.apply_nightly_maintenance({"agent-a"})

    assert result["newly_dead"] == []
    assert world.get_state("agent-a").food == 4
    assert world.get_state("agent-a").water == 4
    assert world.get_state("agent-a").energy >= world.get_state("agent-b").energy


def test_random_event_engine_can_apply_drought():
    """Optional exogenous events should be available without being always-on."""
    world = build_world()
    engine = RandomEventEngine(
        EventConfig(
            enabled=True,
            event_probability=1.0,
            seed=1,
            allowed_events=["drought"],
            drought_water_loss_fraction=0.5,
        )
    )

    events = engine.maybe_apply(world)

    assert events
    assert events[0]["kind"] == "drought"
    assert world.public_water < 10
