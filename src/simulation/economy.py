"""Economy and exchange mechanics for the society simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .world import World


@dataclass
class TradeOffer:
    """A pending trade offer in the world."""

    offer_id: str
    from_agent: str
    to_agent: str
    give_amount: int
    ask_amount: int
    created_at: int
    note: str | None = None
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trade offer."""
        return {
            "offer_id": self.offer_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "give_amount": self.give_amount,
            "ask_amount": self.ask_amount,
            "created_at": self.created_at,
            "note": self.note,
            "status": self.status,
        }


class EconomyEngine:
    """Handle gathering, transfers, theft, and delayed trade settlement."""

    def __init__(self, trade_offer_ttl: int = 3):
        self.trade_offer_ttl = trade_offer_ttl
        self._offer_counter = 0
        self.pending_offers: dict[str, TradeOffer] = {}

    def get_visible_offers(self, agent_id: str) -> list[dict[str, Any]]:
        """Return all open offers addressed to or sent by an agent."""
        offers = []
        for offer in self.pending_offers.values():
            if offer.status != "open":
                continue
            if offer.to_agent == agent_id or offer.from_agent == agent_id:
                offers.append(offer.to_dict())
        return sorted(offers, key=lambda item: item["created_at"])

    def expire_offers(self, world: World) -> list[dict[str, Any]]:
        """Expire stale trade offers."""
        expired_events: list[dict[str, Any]] = []
        for offer in self.pending_offers.values():
            if offer.status != "open":
                continue
            if world.timestep - offer.created_at >= self.trade_offer_ttl:
                offer.status = "expired"
                expired_events.append(
                    world.record_event(
                        kind="trade_expired",
                        actor=offer.from_agent,
                        target=offer.to_agent,
                        public=True,
                        metadata={"offer_id": offer.offer_id},
                    )
                )
        return expired_events

    def _next_offer_id(self) -> str:
        self._offer_counter += 1
        return f"offer-{self._offer_counter}"

    def create_offer(
        self,
        world: World,
        *,
        from_agent: str,
        to_agent: str,
        give_amount: int,
        ask_amount: int,
        note: str | None = None,
        public: bool = True,
    ) -> TradeOffer | None:
        """Create a new trade offer if it is valid."""
        if give_amount < 0 or ask_amount < 0:
            return None
        if from_agent == to_agent or to_agent not in world.agent_states:
            return None
        if world.get_state(from_agent).resources < give_amount:
            return None

        offer = TradeOffer(
            offer_id=self._next_offer_id(),
            from_agent=from_agent,
            to_agent=to_agent,
            give_amount=give_amount,
            ask_amount=ask_amount,
            created_at=world.timestep,
            note=note,
        )
        self.pending_offers[offer.offer_id] = offer
        world.record_event(
            kind="trade_offer",
            actor=from_agent,
            target=to_agent,
            public=public,
            message=note,
            metadata=offer.to_dict(),
        )
        return offer

    def accept_offer(self, world: World, offer_id: str, accepter_id: str) -> tuple[bool, int, dict[str, Any] | None]:
        """Accept a pending trade offer, settling both transfers."""
        offer = self.pending_offers.get(offer_id)
        if not offer or offer.status != "open" or offer.to_agent != accepter_id:
            return (False, 0, None)

        source = world.get_state(offer.from_agent)
        target = world.get_state(offer.to_agent)
        if source.resources < offer.give_amount or target.resources < offer.ask_amount:
            offer.status = "failed"
            event = world.record_event(
                kind="trade_failed",
                actor=offer.from_agent,
                target=offer.to_agent,
                public=True,
                metadata={"offer_id": offer.offer_id},
            )
            return (False, 0, event)

        world.transfer(offer.from_agent, offer.to_agent, offer.give_amount)
        world.transfer(offer.to_agent, offer.from_agent, offer.ask_amount)
        offer.status = "accepted"
        volume = offer.give_amount + offer.ask_amount
        event = world.record_event(
            kind="trade_completed",
            actor=offer.from_agent,
            target=offer.to_agent,
            amount=volume,
            public=True,
            metadata=offer.to_dict(),
        )
        return (True, volume, event)

    def resolve(
        self,
        world: World,
        decisions: dict[str, dict[str, Any]],
        *,
        allow_private_messages: bool,
        allow_steal: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        """Apply a set of agent decisions to the world for one timestep."""
        events = self.expire_offers(world)
        trade_volume = 0

        for agent_id, decision in decisions.items():
            action = str(decision.get("action", "idle"))
            message = str(decision.get("message", "")).strip()
            target = decision.get("target")
            public = not world.get_state(agent_id).unmonitored

            if action == "broadcast" and message:
                world.broadcast(agent_id, message)
                events.append(
                    world.record_event(
                        kind="broadcast",
                        actor=agent_id,
                        message=message,
                        public=True,
                    )
                )
            elif action == "whisper" and message and allow_private_messages and isinstance(target, str):
                if target in world.agent_states:
                    world.whisper(agent_id, target, message)
                    events.append(
                        world.record_event(
                            kind="whisper",
                            actor=agent_id,
                            target=target,
                            message=message,
                            public=False,
                        )
                    )

        for agent_id, decision in decisions.items():
            action = str(decision.get("action", "idle"))
            if action != "accept_trade":
                continue
            offer_id = str(decision.get("offer_id", "")).strip()
            accepted, volume, event = self.accept_offer(world, offer_id, agent_id)
            trade_volume += volume
            if event is not None:
                events.append(event)

        for agent_id, decision in decisions.items():
            action = str(decision.get("action", "idle"))
            target = decision.get("target")
            public = not world.get_state(agent_id).unmonitored

            if action == "gather":
                amount = world.gather(agent_id, int(decision.get("amount", world.config.gather_amount)))
                events.append(
                    world.record_event(
                        kind="gather",
                        actor=agent_id,
                        amount=amount,
                        public=public,
                    )
                )
            elif action == "share" and isinstance(target, str) and target in world.agent_states:
                amount = int(decision.get("amount", 0))
                moved = world.transfer(agent_id, target, amount)
                trade_volume += moved
                events.append(
                    world.record_event(
                        kind="share",
                        actor=agent_id,
                        target=target,
                        amount=moved,
                        public=public,
                    )
                )
            elif action == "offer_trade" and isinstance(target, str) and target in world.agent_states:
                offer = self.create_offer(
                    world,
                    from_agent=agent_id,
                    to_agent=target,
                    give_amount=int(decision.get("give_amount", 0)),
                    ask_amount=int(decision.get("ask_amount", 0)),
                    note=str(decision.get("message", "")).strip() or None,
                    public=public,
                )
                if offer is None:
                    events.append(
                        world.record_event(
                            kind="trade_offer_invalid",
                            actor=agent_id,
                            target=target,
                            public=public,
                        )
                    )
            elif action == "steal" and allow_steal and isinstance(target, str) and target in world.agent_states:
                amount = int(decision.get("amount", world.config.steal_amount))
                stolen = world.steal(agent_id, target, amount)
                trade_volume += stolen
                events.append(
                    world.record_event(
                        kind="steal",
                        actor=agent_id,
                        target=target,
                        amount=stolen,
                        public=public,
                    )
                )

        return (events, trade_volume)
