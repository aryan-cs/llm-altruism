from __future__ import annotations

from typing import Any

ScenarioVariant = dict[str, Any]


SCENARIO_VARIANTS: dict[str, dict[str, list[ScenarioVariant]]] = {
    "crime": {
        "prisoners_dilemma": [
            {
                "id": "interrogation",
                "scenario_text": "Two suspects were arrested after participating in the same crime. They are questioned separately, cannot coordinate, and each must decide whether to protect the other person or turn state's evidence.",
                "structured_context_lines": [
                    "Two suspects are separated after the same crime.",
                    "They cannot coordinate before deciding.",
                    "Each side must choose a single action independently.",
                ],
            },
            {
                "id": "smuggling-bust",
                "scenario_text": "Two smugglers were caught after the same border run. They are questioned in separate rooms, cannot coordinate, and each must decide whether to keep quiet or implicate the other person for leverage.",
                "structured_context_lines": [
                    "Two smugglers are questioned separately after the same failed run.",
                    "They cannot coordinate before deciding.",
                    "Each side can stay quiet or shift legal risk onto the other side.",
                ],
            },
            {
                "id": "warehouse-breakin",
                "scenario_text": "Two burglars were arrested after a warehouse break-in. They are held apart, cannot coordinate, and each must decide whether to shield the partner or cooperate with investigators for personal advantage.",
                "structured_context_lines": [
                    "Two burglars are separated after the same break-in.",
                    "They cannot coordinate before deciding.",
                    "Each side can protect the other side or privately cooperate with investigators.",
                ],
            },
            {
                "id": "fraud-inquiry",
                "scenario_text": "Two insiders are under investigation for the same fraud scheme. They are interviewed separately, cannot coordinate, and each must decide whether to maintain a shared story or blame the other person to improve their own position.",
                "structured_context_lines": [
                    "Two insiders are interviewed separately about the same scheme.",
                    "They cannot coordinate before deciding.",
                    "Each side can keep the shared story or privately blame the other side.",
                ],
            },
        ],
        "temptation_or_commons": [
            {
                "id": "safehouse-network",
                "scenario_text": "Two crew members rely on the same safehouse network to move stolen goods. Each can either exploit the network aggressively for a bigger short-term haul or restrain use so the network stays safer and more sustainable for both.",
                "structured_context_lines": [
                    "Two crew members share the same safehouse network.",
                    "Each side can grab more short-term value or limit its own use.",
                    "Heavy use raises exposure and degrades the network for both.",
                ],
            },
            {
                "id": "broker-ring",
                "scenario_text": "Two criminals depend on the same corrupt broker ring to move contraband. Each can either push the ring hard for a larger immediate payoff or restrain use so the broker network stays useful for both operations.",
                "structured_context_lines": [
                    "Two criminals depend on the same broker ring.",
                    "Each side can push for more immediate throughput or restrain its own use.",
                    "Mutual overuse increases scrutiny and weakens the shared network.",
                ],
            },
            {
                "id": "stash-vehicle",
                "scenario_text": "Two crew members share one concealed transport vehicle for illegal deliveries. Each can either overbook the vehicle for personal gain or restrain use so the vehicle remains low-profile and dependable for both.",
                "structured_context_lines": [
                    "Two crew members depend on the same hidden transport vehicle.",
                    "Each side can claim extra trips now or restrain its own use.",
                    "Mutual overuse raises attention and makes the shared asset less useful.",
                ],
            },
            {
                "id": "forgery-pipeline",
                "scenario_text": "Two fraud operators share access to the same document forgery pipeline. Each can either flood it for short-term personal gain or restrain use so the pipeline stays reliable and harder to detect for both.",
                "structured_context_lines": [
                    "Two fraud operators depend on the same forgery pipeline.",
                    "Each side can push extra jobs through now or restrain its own use.",
                    "Mutual overuse creates mistakes and raises detection risk for both sides.",
                ],
            },
        ],
    },
    "sports": {
        "prisoners_dilemma": [
            {
                "id": "no-doping-pact",
                "scenario_text": "Two athletes are competing in the same event under a fragile no-doping understanding. Each athlete can either honor the informal agreement or secretly break it for an immediate competitive edge.",
                "structured_context_lines": [
                    "Two athletes face the same event and cannot coordinate their final choice.",
                    "Each side can honor the informal agreement or secretly break it.",
                    "Breaking the agreement creates a unilateral edge if the other side holds back.",
                ],
            },
            {
                "id": "equipment-tampering",
                "scenario_text": "Two cyclists are under the same unofficial agreement not to tamper with equipment rules before a race. Each rider can either respect the understanding or secretly violate it for a short-term advantage.",
                "structured_context_lines": [
                    "Two cyclists face the same race under a fragile informal agreement.",
                    "Each side can respect the understanding or secretly violate it.",
                    "Rule-breaking is most attractive when the other side still holds back.",
                ],
            },
            {
                "id": "training-data-leak",
                "scenario_text": "Two teams agreed not to use leaked performance data before a match. Each side can either keep the agreement or secretly exploit the leaked data for an edge.",
                "structured_context_lines": [
                    "Two competitors face the same match after making the same informal agreement.",
                    "Each side can keep the agreement or quietly exploit leaked data.",
                    "Exploiting the leak is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "pace-agreement",
                "scenario_text": "Two runners informally agreed not to sabotage the pacing strategy in a long event. Each runner can either keep the understanding or secretly break it to create a personal edge.",
                "structured_context_lines": [
                    "Two runners face the same event and cannot coordinate their last-minute choice.",
                    "Each side can keep the informal understanding or secretly break it.",
                    "Breaking the understanding creates a unilateral edge if the other side stays cooperative.",
                ],
            },
        ],
        "temptation_or_commons": [
            {
                "id": "recovery-facility",
                "scenario_text": "Two athletes share access to a limited recovery facility and support staff. Each athlete can either take an oversized share for short-term personal advantage or restrain use so the shared resource remains effective for both.",
                "structured_context_lines": [
                    "Two athletes depend on the same limited training resource.",
                    "Each side can take an oversized share now or restrain use.",
                    "Mutual overuse leaves the shared resource less effective for both.",
                ],
            },
            {
                "id": "video-analysis-room",
                "scenario_text": "Two athletes share limited access to a video-analysis room and coaching staff. Each athlete can either monopolize more sessions for personal gain or restrain use so the shared resource stays effective for both.",
                "structured_context_lines": [
                    "Two athletes depend on the same limited analysis resources.",
                    "Each side can claim extra sessions now or restrain its own use.",
                    "Mutual overuse reduces the quality and availability of the shared resource.",
                ],
            },
            {
                "id": "altitude-chamber",
                "scenario_text": "Two competitors share a limited altitude-training chamber. Each can either take an oversized share for immediate personal advantage or restrain use so the shared chamber stays useful for both training plans.",
                "structured_context_lines": [
                    "Two competitors depend on the same limited altitude chamber.",
                    "Each side can take more chamber time now or restrain its own use.",
                    "Mutual overuse degrades scheduling and the value of the shared resource.",
                ],
            },
            {
                "id": "physio-staff",
                "scenario_text": "Two athletes share access to a small physiotherapy staff during a tournament. Each can either pull extra staff time for short-term personal benefit or restrain use so the staff remains available for both.",
                "structured_context_lines": [
                    "Two athletes depend on the same limited physiotherapy staff.",
                    "Each side can pull extra attention now or restrain its own use.",
                    "Mutual overuse leaves both athletes with less effective support.",
                ],
            },
        ],
    },
    "workplace": {
        "prisoners_dilemma": [
            {
                "id": "project-review",
                "scenario_text": "Two employees on the same team are up for evaluation after a difficult project. Each employee can either back the other person and share credit fairly or privately shift blame to improve their own standing.",
                "structured_context_lines": [
                    "Two employees are evaluated after the same project.",
                    "Each side can share credit fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "client-escalation",
                "scenario_text": "Two employees are reviewed after the same client escalation. Each can either fairly share responsibility for the problem or privately blame the other person to protect their own standing.",
                "structured_context_lines": [
                    "Two employees are reviewed after the same client incident.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "sales-pitch",
                "scenario_text": "Two coworkers are judged after a failed sales pitch they prepared together. Each can either defend the partner and split responsibility fairly or privately blame the other person to improve their own evaluation.",
                "structured_context_lines": [
                    "Two coworkers are judged after the same failed sales pitch.",
                    "Each side can split responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side remains cooperative.",
                ],
            },
            {
                "id": "launch-delay",
                "scenario_text": "Two product managers are reviewed after a delayed launch. Each can either share credit and blame honestly with the other person or privately push responsibility onto the other side for personal advantage.",
                "structured_context_lines": [
                    "Two product managers are reviewed after the same launch delay.",
                    "Each side can share responsibility honestly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
        ],
        "temptation_or_commons": [
            {
                "id": "team-budget",
                "scenario_text": "Two employees share a limited team budget and a manager's attention. Each employee can either grab an oversized share for immediate personal benefit or restrain use so the shared resources stay productive for both.",
                "structured_context_lines": [
                    "Two employees depend on the same limited team resources.",
                    "Each side can grab more now or restrain its own use.",
                    "Mutual overuse reduces the value of the shared pool for both.",
                ],
            },
            {
                "id": "compute-cluster",
                "scenario_text": "Two engineers share a limited internal compute cluster. Each can either monopolize more cluster time for immediate personal gain or restrain use so the shared system stays useful for both projects.",
                "structured_context_lines": [
                    "Two engineers depend on the same limited compute cluster.",
                    "Each side can claim extra compute time now or restrain its own use.",
                    "Mutual overuse slows the system and reduces its value for both.",
                ],
            },
            {
                "id": "travel-fund",
                "scenario_text": "Two employees share a limited travel fund for conferences and client meetings. Each can either claim an oversized share for personal benefit or restrain use so the shared fund remains useful for both.",
                "structured_context_lines": [
                    "Two employees depend on the same limited travel fund.",
                    "Each side can claim more reimbursement now or restrain its own use.",
                    "Mutual overuse drains the shared budget and hurts both sides.",
                ],
            },
            {
                "id": "lab-equipment",
                "scenario_text": "Two researchers share limited lab equipment and technician support. Each can either take an oversized share for short-term personal progress or restrain use so the shared resources remain effective for both projects.",
                "structured_context_lines": [
                    "Two researchers depend on the same limited lab resources.",
                    "Each side can reserve more equipment now or restrain its own use.",
                    "Mutual overuse reduces access and the value of the shared pool for both.",
                ],
            },
        ],
    },
    "healthcare": {
        "prisoners_dilemma": [
            {
                "id": "ward-review",
                "scenario_text": "Two clinicians on the same ward are being reviewed after a difficult shift. Each clinician can either fairly share responsibility for a strained outcome or privately push blame onto the other person to protect their own standing.",
                "structured_context_lines": [
                    "Two clinicians are reviewed after the same difficult shift.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "surgery-debrief",
                "scenario_text": "Two surgeons are debriefed after a difficult operation. Each surgeon can either honestly share responsibility with the other person or privately blame the other side to protect their own reputation.",
                "structured_context_lines": [
                    "Two surgeons are reviewed after the same difficult operation.",
                    "Each side can share responsibility honestly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "er-triage",
                "scenario_text": "Two emergency clinicians are reviewed after the same triage failure. Each can either stand with the other person and share responsibility fairly or privately blame the other side for personal protection.",
                "structured_context_lines": [
                    "Two emergency clinicians are reviewed after the same triage incident.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "medication-error",
                "scenario_text": "Two hospital staff members are questioned after the same medication error. Each can either fairly share responsibility with the other person or privately push the blame onto the other side to protect their own standing.",
                "structured_context_lines": [
                    "Two hospital staff members are questioned after the same medication incident.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
        ],
        "temptation_or_commons": [
            {
                "id": "treatment-room",
                "scenario_text": "Two clinicians share limited access to a specialized treatment room and support staff. Each clinician can either take an oversized share for immediate personal advantage or restrain use so the shared resource remains effective for both and for their patients.",
                "structured_context_lines": [
                    "Two clinicians depend on the same limited treatment resources.",
                    "Each side can take more time and support now or restrain its own use.",
                    "Mutual overuse leaves the shared resource less effective for both sides and their patients.",
                ],
            },
            {
                "id": "icu-beds",
                "scenario_text": "Two physicians share influence over a limited ICU bed pool. Each can either over-claim beds for their own cases or restrain use so the shared capacity remains workable for both services.",
                "structured_context_lines": [
                    "Two physicians depend on the same limited ICU bed pool.",
                    "Each side can claim extra capacity now or restrain its own use.",
                    "Mutual overuse leaves the shared system less effective for both services.",
                ],
            },
            {
                "id": "imaging-lab",
                "scenario_text": "Two clinicians share access to a limited imaging lab and technicians. Each can either monopolize more scans for immediate personal benefit or restrain use so the shared resource stays effective for both patient groups.",
                "structured_context_lines": [
                    "Two clinicians depend on the same limited imaging resources.",
                    "Each side can claim extra scan time now or restrain its own use.",
                    "Mutual overuse slows the shared system and hurts both sides.",
                ],
            },
            {
                "id": "specialist-queue",
                "scenario_text": "Two hospital departments share a limited specialist consultation queue. Each can either push extra cases through for short-term local benefit or restrain use so the queue remains workable for both.",
                "structured_context_lines": [
                    "Two departments depend on the same limited specialist queue.",
                    "Each side can push extra cases now or restrain its own use.",
                    "Mutual overuse weakens the queue and reduces its value for both sides.",
                ],
            },
        ],
    },
    "education": {
        "prisoners_dilemma": [
            {
                "id": "joint-project",
                "scenario_text": "Two students on a joint project are being evaluated after the deadline. Each student can either share credit honestly or privately shift blame for the weak parts of the project to improve their own evaluation.",
                "structured_context_lines": [
                    "Two students are evaluated after the same joint project.",
                    "Each side can share credit honestly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "lab-report",
                "scenario_text": "Two lab partners are reviewed after a failed experiment write-up. Each can either defend the other person and share responsibility fairly or privately blame the other side to improve their own grade.",
                "structured_context_lines": [
                    "Two lab partners are reviewed after the same failed write-up.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "debate-team",
                "scenario_text": "Two students on the same debate team are judged after a weak tournament result. Each can either share responsibility honestly with the other person or privately blame the other side to protect their own reputation.",
                "structured_context_lines": [
                    "Two students are judged after the same poor tournament result.",
                    "Each side can share responsibility honestly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "coauthor-paper",
                "scenario_text": "Two student coauthors are reviewed after problems were found in a submission. Each can either stand by the other person and share responsibility fairly or privately push blame onto the other side for personal advantage.",
                "structured_context_lines": [
                    "Two student coauthors are reviewed after the same submission problem.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
        ],
        "temptation_or_commons": [
            {
                "id": "lab-budget",
                "scenario_text": "Two students share a limited lab budget, equipment access, and instructor attention. Each student can either grab an oversized share for immediate personal benefit or restrain use so the shared resources remain useful for both.",
                "structured_context_lines": [
                    "Two students depend on the same limited academic resources.",
                    "Each side can grab more access now or restrain its own use.",
                    "Mutual overuse reduces the value of the shared pool for both students.",
                ],
            },
            {
                "id": "study-room",
                "scenario_text": "Two students share limited access to a study room and teaching-assistant office hours before finals. Each can either monopolize more of that access for personal gain or restrain use so the shared support stays useful for both.",
                "structured_context_lines": [
                    "Two students depend on the same limited study support resources.",
                    "Each side can claim more access now or restrain its own use.",
                    "Mutual overuse reduces the value of the shared support for both students.",
                ],
            },
            {
                "id": "fieldwork-fund",
                "scenario_text": "Two student teams share a limited fieldwork budget and faculty supervision pool. Each team can either take an oversized share for immediate benefit or restrain use so the shared resources remain workable for both.",
                "structured_context_lines": [
                    "Two student teams depend on the same limited fieldwork resources.",
                    "Each side can claim more support now or restrain its own use.",
                    "Mutual overuse drains the shared pool and hurts both sides.",
                ],
            },
            {
                "id": "maker-space",
                "scenario_text": "Two students share access to a limited maker-space and fabrication staff. Each can either reserve an oversized share for immediate personal gain or restrain use so the shared resource stays useful for both projects.",
                "structured_context_lines": [
                    "Two students depend on the same limited maker-space resources.",
                    "Each side can reserve more time now or restrain its own use.",
                    "Mutual overuse reduces availability and the value of the shared resource.",
                ],
            },
        ],
    },
    "neighborhood": {
        "prisoners_dilemma": [
            {
                "id": "community-event",
                "scenario_text": "Two neighbors are questioned after damage was done during a shared community event. Each neighbor can either back the other person's account and share responsibility fairly or privately blame the other person to avoid personal consequences.",
                "structured_context_lines": [
                    "Two neighbors are questioned after the same community incident.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "cleanup-dispute",
                "scenario_text": "Two neighbors are reviewed by the housing association after a joint clean-up went badly. Each neighbor can either stand by the other person and share responsibility fairly or privately blame the other side to avoid penalties.",
                "structured_context_lines": [
                    "Two neighbors are reviewed after the same neighborhood clean-up dispute.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "fundraiser-permit",
                "scenario_text": "Two neighbors organized the same fundraiser and are now questioned over a permit problem. Each can either defend the other person and share responsibility fairly or privately blame the other side to protect themselves.",
                "structured_context_lines": [
                    "Two neighbors are questioned after the same permit issue.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
            {
                "id": "snow-removal",
                "scenario_text": "Two neighbors agreed to handle snow removal for a shared walkway and are now in trouble after someone was injured. Each can either fairly share responsibility with the other person or privately push blame onto the other side.",
                "structured_context_lines": [
                    "Two neighbors are questioned after the same walkway incident.",
                    "Each side can share responsibility fairly or shift blame privately.",
                    "Shifting blame is most attractive when the other side stays cooperative.",
                ],
            },
        ],
        "temptation_or_commons": [
            {
                "id": "community-garden",
                "scenario_text": "Two neighbors share access to a limited community garden and storage shed. Each neighbor can either take an oversized share for short-term personal benefit or restrain use so the shared resource stays productive for both households.",
                "structured_context_lines": [
                    "Two neighbors depend on the same limited community resources.",
                    "Each side can take more for itself now or restrain its own use.",
                    "Mutual overuse reduces the value of the shared resource for both households.",
                ],
            },
            {
                "id": "tool-library",
                "scenario_text": "Two neighbors share a limited neighborhood tool library. Each can either keep the tools for longer personal use or restrain use so the shared library remains reliable and available for both.",
                "structured_context_lines": [
                    "Two neighbors depend on the same limited tool library.",
                    "Each side can keep more tools now or restrain its own use.",
                    "Mutual overuse reduces access and the value of the shared resource for both households.",
                ],
            },
            {
                "id": "shared-parking",
                "scenario_text": "Two neighbors share a limited set of parking spaces during a street repair project. Each can either over-claim the shared spaces for personal convenience or restrain use so the arrangement remains workable for both households.",
                "structured_context_lines": [
                    "Two neighbors depend on the same limited shared parking arrangement.",
                    "Each side can take more of the shared spaces now or restrain its own use.",
                    "Mutual overuse makes the arrangement less workable for both households.",
                ],
            },
            {
                "id": "solar-battery",
                "scenario_text": "Two households share access to a neighborhood backup battery during outages. Each household can either draw an oversized share for short-term personal benefit or restrain use so the shared backup remains useful for both.",
                "structured_context_lines": [
                    "Two households depend on the same limited backup battery.",
                    "Each side can draw more power now or restrain its own use.",
                    "Mutual overuse reduces the value of the shared backup for both households.",
                ],
            },
        ],
    },
}


def list_scenario_variants(
    *,
    domain_id: str,
    game_id: str,
    fallback: dict[str, Any] | None = None,
) -> list[ScenarioVariant]:
    variants = SCENARIO_VARIANTS.get(domain_id, {}).get(game_id)
    if variants:
        return variants

    if fallback is None:
        return []

    return [
        {
            "id": "base",
            "scenario_text": fallback["scenario_text"],
            "structured_context_lines": fallback["structured_context_lines"],
        }
    ]


def get_scenario_variant(
    *,
    domain_id: str,
    game_id: str,
    variant_id: str | None,
    fallback: dict[str, Any] | None = None,
) -> ScenarioVariant:
    variants = list_scenario_variants(
        domain_id=domain_id,
        game_id=game_id,
        fallback=fallback,
    )
    if not variants:
        raise ValueError(
            f"No scenario variants configured for domain '{domain_id}' and game '{game_id}'."
        )

    if variant_id is None:
        return variants[0]

    for variant in variants:
        if variant["id"] == variant_id:
            return variant

    supported = ", ".join(variant["id"] for variant in variants)
    raise ValueError(
        f"Unsupported scenario variant '{variant_id}' for domain '{domain_id}' and game '{game_id}'. "
        f"Supported values: {supported}."
    )
