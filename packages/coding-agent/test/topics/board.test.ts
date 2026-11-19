/**
 * Contract: the topic board lists one row per topic (title, id, state, last
 * activity, gist), open topics before closed ones, newest activity first, with
 * rows past the cap reduced to title and id; a session gets a new board only
 * when its text differs from the latest board in its history.
 */
import { describe, expect, it } from "bun:test";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import {
	formatTopicBoard,
	TOPIC_BOARD_CONTEXT_TYPE,
	TOPIC_BOARD_FULL_ROWS,
	topicBoardMessage,
} from "@oh-my-pi/pi-coding-agent/topics/board";
import type { WorkerSummary } from "@oh-my-pi/pi-coding-agent/vibe/runtime";

const T0 = Date.UTC(2026, 10, 19, 7, 5, 30);

function topic(id: string, overrides: Partial<WorkerSummary> = {}): WorkerSummary {
	return {
		id,
		title: `${id} title`,
		state: "idle",
		turns: 1,
		queued: 0,
		createdAt: T0,
		lastActivityAt: T0,
		...overrides,
	};
}

function rows(board: string | undefined): string[] {
	return (board ?? "").split("\n").filter(line => line.startsWith("- "));
}

describe("formatTopicBoard", () => {
	it("is absent without topics", () => {
		expect(formatTopicBoard([], "inbox")).toBeUndefined();
	});

	it("renders title, id, state, minute UTC time and a one-line gist per topic, with how to act", () => {
		const board = formatTopicBoard(
			[
				topic("Diokan", {
					title: "Diokan <term> sheet",
					state: "running",
					queued: 2,
					lastActivity: "Drafting clause\n4",
				}),
				topic("Notes", { lastActivityAt: T0 - 60_000 }),
				topic("Starting", { state: "starting", lastActivityAt: T0 - 120_000 }),
				topic("Old", { state: "closed", lastActivityAt: T0 + 60_000 }),
			],
			"inbox",
		);
		expect(rows(board)).toEqual([
			"- Diokan &lt;term&gt; sheet (id: Diokan): working, queued 2; last activity 2026-11-19T07:05Z; Drafting clause 4",
			"- Notes title (id: Notes): idle; last activity 2026-11-19T07:04Z",
			"- Starting title (id: Starting): working; last activity 2026-11-19T07:03Z",
			"- Old title (id: Old): closed; last activity 2026-11-19T07:06Z",
		]);
		expect(board).toStartWith("<topic-board>");
		expect(board).toEndWith("</topic-board>");
		expect(board).toContain("`read history://<id>`");
		expect(board).toContain("`agent://<id>`");
	});

	it("lists open topics newest first, then closed ones newest first", () => {
		const board = formatTopicBoard(
			[
				topic("ClosedOld", { state: "closed", lastActivityAt: T0 - 5 * 60_000 }),
				topic("OpenOld", { lastActivityAt: T0 - 4 * 60_000 }),
				topic("ClosedNew", { state: "closed", lastActivityAt: T0 + 60 * 60_000 }),
				topic("OpenNew", { state: "running", lastActivityAt: T0 }),
			],
			"inbox",
		);
		expect(rows(board).map(row => row.match(/\(id: (\w+)\)/)?.[1])).toEqual([
			"OpenNew",
			"OpenOld",
			"ClosedNew",
			"ClosedOld",
		]);
	});

	it(`keeps ${TOPIC_BOARD_FULL_ROWS} full rows and lists the rest by title and id only`, () => {
		const topics = Array.from({ length: TOPIC_BOARD_FULL_ROWS + 5 }, (_, index) =>
			topic(`T${index}`, { lastActivityAt: T0 - index * 60_000, lastActivity: `gist ${index}` }),
		);
		topics.push(topic("Shut", { state: "closed", lastActivityAt: T0 + 60_000 }));
		const board = formatTopicBoard(topics, "inbox");
		const listed = rows(board);
		expect(listed).toHaveLength(TOPIC_BOARD_FULL_ROWS + 6);
		expect(listed[TOPIC_BOARD_FULL_ROWS - 1]).toContain(`(id: T${TOPIC_BOARD_FULL_ROWS - 1}): idle; last activity`);
		expect(listed[TOPIC_BOARD_FULL_ROWS]).toBe(`- T${TOPIC_BOARD_FULL_ROWS} title (id: T${TOPIC_BOARD_FULL_ROWS})`);
		// The closed topic is newer than every open one but still comes last, beyond the cap.
		expect(listed.at(-1)).toBe("- Shut title (id: Shut)");
		expect(board).toContain(`past the first ${TOPIC_BOARD_FULL_ROWS}, titles only`);
	});

	it("tells a topic the board lists its siblings", () => {
		const board = formatTopicBoard([topic("Notes")], "topic");
		expect(board).toContain("You are a topic too; you are not listed.");
		expect(formatTopicBoard([topic("Notes")], "inbox")).not.toContain("You are a topic too");
	});
});

describe("topicBoardMessage", () => {
	const user = (text: string): AgentMessage => ({ role: "user", content: text, timestamp: T0 });
	const board = (content: string): AgentMessage => ({
		role: "custom",
		customType: TOPIC_BOARD_CONTEXT_TYPE,
		content,
		display: false,
		timestamp: T0,
	});

	it("sends a board the history has not seen yet, hidden and agent-attributed", () => {
		expect(topicBoardMessage("A", [user("hi")])).toMatchObject({
			role: "custom",
			customType: TOPIC_BOARD_CONTEXT_TYPE,
			content: "A",
			display: false,
			attribution: "agent",
		});
	});

	it("sends nothing when the latest board in the history is the same text", () => {
		expect(topicBoardMessage("A", [board("A"), user("hi")])).toBeUndefined();
		expect(topicBoardMessage(undefined, [user("hi")])).toBeUndefined();
	});

	it("compares against the latest board only", () => {
		expect(topicBoardMessage("A", [board("A"), user("one"), board("B"), user("two")])?.content).toBe("A");
	});
});
