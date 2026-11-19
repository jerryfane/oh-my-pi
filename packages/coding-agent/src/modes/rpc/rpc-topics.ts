/**
 * RPC `topics_*` commands: the host's direct line to the owner's topics. Every
 * command acts on the topic itself; none starts a turn of the main session.
 */
import type { TopicsHost } from "../../topics/host";
import type { RpcCommand, RpcResponse, RpcSubagentMessagesResult } from "./rpc-types";

/** Owner-facing topics (persistent workers the host talks to directly); part of `RpcCommand`. */
export type RpcTopicsCommand =
	| { id?: string; type: "topics_list" }
	| { id?: string; type: "topics_new"; title: string; message: string }
	| { id?: string; type: "topics_send"; topicId: string; message: string }
	| { id?: string; type: "topics_read"; topicId: string; fromByte?: number }
	| { id?: string; type: "topics_close"; topicId: string }
	| { id?: string; type: "topics_reopen"; topicId: string }
	| { id?: string; type: "topics_rename"; topicId: string; title: string };

/** One owner-facing topic (persistent worker). */
export interface RpcTopicSummary {
	id: string;
	title: string;
	state: "starting" | "running" | "idle" | "dead" | "closed";
	turns: number;
	queued: number;
	createdAt: number;
	lastActivityAt: number;
	lastActivity?: string;
	/** First line of the owner's latest message to the topic, when this process saw one. */
	ownerLine?: string;
	sessionFile?: string;
	/** When the turn in flight started; absent while no turn runs. */
	turnStartedAt?: number;
}

/** Successful responses to {@link RpcTopicsCommand}; part of `RpcResponse`. */
export type RpcTopicsResponse =
	| { id?: string; type: "response"; command: "topics_list"; success: true; data: { topics: RpcTopicSummary[] } }
	| { id?: string; type: "response"; command: "topics_new"; success: true; data: { topicId: string } }
	| {
			id?: string;
			type: "response";
			command: "topics_send";
			success: true;
			data: { topicId: string; mode: "steered" | "queued" | "turn" };
	  }
	| { id?: string; type: "response"; command: "topics_read"; success: true; data: RpcSubagentMessagesResult }
	| { id?: string; type: "response"; command: "topics_close"; success: true }
	| { id?: string; type: "response"; command: "topics_reopen"; success: true }
	| { id?: string; type: "response"; command: "topics_rename"; success: true };

/** Whether `command` is a `topics_*` command. */
export function isRpcTopicsCommand(command: RpcCommand): command is RpcTopicsCommand {
	return command.type.startsWith("topics_");
}

function requireText(value: unknown, field: string): string {
	if (typeof value !== "string" || !value.trim()) throw new Error(`\`${field}\` must be a non-empty string.`);
	return value;
}

/** Run one `topics_*` command; returns the response `data` (undefined = none). Throws a user-facing error. */
export async function runRpcTopicsCommand(host: TopicsHost, command: RpcTopicsCommand): Promise<object | undefined> {
	switch (command.type) {
		case "topics_list":
			return { topics: host.list() };
		case "topics_new": {
			const { id } = await host.create({
				title: requireText(command.title, "title"),
				message: requireText(command.message, "message"),
			});
			return { topicId: id };
		}
		case "topics_send": {
			const outcome = await host.send(
				requireText(command.topicId, "topicId"),
				requireText(command.message, "message"),
			);
			return { topicId: outcome.id, mode: outcome.mode };
		}
		case "topics_read":
			if (command.fromByte !== undefined && !Number.isFinite(command.fromByte)) {
				throw new Error("`fromByte` must be a finite number.");
			}
			return host.read(requireText(command.topicId, "topicId"), command.fromByte);
		case "topics_close":
			await host.close(requireText(command.topicId, "topicId"));
			return undefined;
		case "topics_reopen":
			await host.reopen(requireText(command.topicId, "topicId"));
			return undefined;
		case "topics_rename":
			await host.rename(requireText(command.topicId, "topicId"), requireText(command.title, "title"));
			return undefined;
	}
}

/** The RPC response to a `topics_*` command, run on the session's topics host (`undefined` = topics disabled). */
export async function respondRpcTopicsCommand(
	topics: Promise<TopicsHost | undefined>,
	command: RpcTopicsCommand,
): Promise<RpcResponse> {
	const { id, type } = command;
	const fail = (error: string): RpcResponse => ({ id, type: "response", command: type, success: false, error });
	const host = await topics;
	if (!host) return fail("Topics are disabled; enable the topics.enabled setting.");
	try {
		const data = await runRpcTopicsCommand(host, command);
		return (
			data === undefined
				? { id, type: "response", command: type, success: true }
				: { id, type: "response", command: type, success: true, data }
		) as RpcResponse;
	} catch (err) {
		return fail(err instanceof Error ? err.message : String(err));
	}
}
