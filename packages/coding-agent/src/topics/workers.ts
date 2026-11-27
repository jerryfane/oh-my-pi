/**
 * The topics registry ({@link topicsRegistry}): a {@link VibeSessionRegistry}
 * of owner-facing workers, the policy it runs them under (sibling board,
 * answers to asks), and how an agent's IRC message to a topic becomes a topic turn.
 */
import type { IrcDeliveryReceipt, IrcMessage } from "../irc/irc-types";
import { logger, prompt } from "@oh-my-pi/pi-utils";
import type { IrcBusApi } from "../irc/bus";
import { IrcBus } from "../irc/bus";
import type { IrcRouteDecision } from "../irc/routes";
import topicAskTemplate from "../prompts/system/topic-ask.md" with { type: "text" };
import { AgentRegistry } from "../registry/agent-registry";
import { escapeHarnessTags } from "../session/harness-tags";
import { buildWakeRelayBody } from "../task/executor";
import type { ToolSession } from "../tools";
import { VibeSessionRegistry } from "../vibe/runtime";
import type { QueuedMessage, SettledWorkerTurn, WorkerScopePolicy, WorkerView } from "../vibe/worker-scope";
import { formatTopicBoard } from "./board";

/**
 * Send a settled turn's outcome to the agents whose asks started it, as the
 * IRC wake-turn relay words it ({@link buildWakeRelayBody}). Always sent,
 * even after the topic messaged the asker mid-turn (that may have been a
 * question back): the topic is told its answer goes back when the turn
 * ends. Relays are answers, so they never draw another relay.
 */
async function answerAskers({ id, askers, result, error }: SettledWorkerTurn): Promise<void> {
	if (askers.length === 0) return;
	const bus = IrcBus.global();
	const aborted = result?.aborted === true;
	const failure = error ?? (result && result.exitCode !== 0 && !aborted ? (result.error ?? "turn failed") : undefined);
	const body = buildWakeRelayBody({
		id,
		yielded: false,
		result,
		turnText: result?.output ?? "",
		error: failure,
		aborted,
		abortReason: result?.abortReason,
		finalizeError: undefined,
		alreadyMessaged: false,
	});
	for (const asker of askers) {
		// The turn already settled: a failed hand-off is logged, never turned into a turn failure.
		const receipt = await bus
			.send({ from: id, to: asker.from, body, replyTo: asker.messageId, wakeRelay: true })
			.catch((sendError: unknown): IrcDeliveryReceipt => ({
				to: asker.from,
				outcome: "failed",
				error: sendError instanceof Error ? sendError.message : String(sendError),
			}));
		if (receipt.outcome === "failed") {
			logger.warn("topics: answer to an ask was not delivered", { from: id, to: asker.from, error: receipt.error });
		}
	}
}

/**
 * Owner-facing, permanent workers whose results the owner reads in the topic
 * itself: run at the owner's spawn depth, shown their siblings' board ahead
 * of each turn, and answering the agents whose asks started a turn.
 */
export const TOPICS_POLICY: WorkerScopePolicy = {
	scope: "topics",
	deliverResults: false,
	ownerSpawnDepth: true,
	boardFor: siblings => formatTopicBoard(siblings, "topic"),
	onTurnSettled: answerAskers,
};

let globalTopics: VibeSessionRegistry | undefined;

/** The process-global registry of owner-facing topic workers, run under {@link TOPICS_POLICY}. */
export function topicsRegistry(): VibeSessionRegistry {
	globalTopics ??= new VibeSessionRegistry(TOPICS_POLICY);
	return globalTopics;
}

/** Drop the topics registry, as a process restart would. Test-only. */
export function resetTopicsRegistryForTests(): void {
	globalTopics = undefined;
}

/** The live topic `agentId` is, or the nearest one it works for (its spawn ancestry). */
function topicOf(registry: VibeSessionRegistry, session: ToolSession, agentId: string): WorkerView | undefined {
	const agents = AgentRegistry.global();
	const seen = new Set<string>();
	for (let id: string | undefined = agentId; id !== undefined && !seen.has(id); id = agents.get(id)?.parentId) {
		const topic = registry.worker(session, id);
		if (topic) return topic;
		seen.add(id);
	}
	return undefined;
}

/**
 * Take an agent's IRC message to one of `session`'s topics into the topics
 * registry (an `IrcBus` route; see `topics/host`), so the turn it starts gets
 * the sibling board and counts like any other, labelled with the sender. Kinds:
 *
 * - ask: runs in a turn of its own (queued behind a busy topic), whose
 *   outcome goes back to the asker when it settles ({@link answerAskers});
 * - reply (that settled outcome, or a non-topic agent answering what the
 *   topic asked it): starts a turn when idle, but no answer goes back;
 * - a topic writing to a topic whose ask started its turn (a question back,
 *   an early answer), or progress: rides along with the next turn, never starts one.
 *
 * Enforced here, not by prompt: a closed topic refuses all of it, and a
 * topic (or a helper it runs) asks another topic only in a turn its owner
 * started (no chains), so topics never drive each other in a loop. In a
 * turn another agent's message started, a progress note never starts a turn
 * anywhere: to a non-topic recipient it goes with `noWake`. A reply or
 * progress to a streaming turn, and anything from the topic's own helpers,
 * is left to ordinary IRC delivery, which hands it into that turn. Returns
 * `undefined` for anything else: no topic of `session` sent or receives it.
 */
export function routeTopicMessage(
	registry: VibeSessionRegistry,
	session: ToolSession,
	message: IrcMessage,
	bus: Pick<IrcBusApi, "unansweredSends">,
): IrcRouteDecision {
	if (!session.getSessionId?.()) return undefined;
	// The sender's topic: the sender itself, or the topic it works for.
	const asking = topicOf(registry, session, message.from);
	const startedBy = asking?.turn?.startedBy ?? [];
	const topic = registry.worker(session, message.to);
	if (!topic) {
		return message.progress === true && startedBy.length > 0 ? { message: { ...message, noWake: true } } : undefined;
	}
	if (asking?.id === topic.id) return undefined;
	const failed = (error: string) => ({ receipt: { to: topic.id, outcome: "failed" as const, error } });
	if (topic.closed) return failed(`Topic "${topic.id}" is closed; its owner has to reopen it first.`);
	const rideAlong = message.progress === true || asking?.turn?.askers.some(asker => asker.from === topic.id) === true;
	const reply =
		message.wakeRelay === true || (!asking && bus.unansweredSends(topic.id, 0).some(row => row.to === message.from));
	if ((rideAlong || reply) && topic.turn?.streaming) return undefined;
	if (!rideAlong && !reply && asking && (!asking.turn || startedBy.length > 0)) {
		const who =
			asking.id === message.from ? `Topic "${asking.id}"` : `"${message.from}", working for topic "${asking.id}",`;
		return failed(
			startedBy.length > 0
				? `${who} cannot ask topic "${topic.id}": the current turn was started by a message from ${startedBy.join(", ")}, and a turn another agent started cannot ask other topics (no chains). Its answer goes back to ${startedBy.join(", ")} when the turn ends.`
				: `${who} cannot ask topic "${topic.id}" outside a turn its owner started (no chains).`,
		);
	}
	const sender = message.from === topic.ownerId ? `Inbox (\`${message.from}\`)` : `\`${message.from}\``;
	const ask = !rideAlong && !reply && message.broadcast !== true;
	const label =
		message.progress === true
			? `Progress from ${sender}`
			: rideAlong || reply
				? `Reply from ${sender}`
				: message.broadcast
					? `Broadcast by ${sender}`
					: `Asked by ${sender}`;
	const entry: QueuedMessage = {
		text: prompt
			.render(topicAskTemplate, {
				label,
				message: escapeHarnessTags(message.body),
				from: message.from,
				relay: ask,
				note: rideAlong,
			})
			.trim(),
		from: message.from,
		...(ask ? { askId: message.id } : {}),
		...(rideAlong ? { note: true } : {}),
	};
	try {
		const started = registry.enqueue(session, topic.id, entry);
		return { receipt: { to: topic.id, outcome: started ? "woken" : "queued" } };
	} catch (error) {
		return failed(error instanceof Error ? error.message : String(error));
	}
}
