/**
 * Shared subagent output artifact persistence (`<id>.md` + optional `<id>.json` sidecar).
 *
 * Kept in a leaf module so both the native subagent executor and external runners
 * (such as `agy-cli-runner`) share the exact same atomic publication and stale-sidecar
 * cleanup behavior without introducing an import cycle.
 */

import * as fs from "node:fs/promises";
import path from "node:path";
import { logger } from "@oh-my-pi/pi-utils";
import { writeArtifact } from "../session/artifacts";
import type { StructuredSubagentOutput } from "./task-details";

export async function persistSubagentOutputArtifacts(args: {
	artifactsDir: string;
	id: string;
	rawOutput: string;
	structuredOutput?: StructuredSubagentOutput;
}): Promise<{ outputPath?: string; outputMeta?: { lineCount: number; charCount: number } }> {
	const { artifactsDir, id, rawOutput, structuredOutput } = args;
	let outputPath: string | undefined;
	let outputMeta: { lineCount: number; charCount: number } | undefined;
	const candidatePath = path.join(artifactsDir, `${id}.md`);
	try {
		await writeArtifact(candidatePath, rawOutput);
		outputPath = candidatePath;
		outputMeta = {
			lineCount: rawOutput.split("\n").length,
			charCount: rawOutput.length,
		};
	} catch (error) {
		logger.warn("Failed to persist subagent output artifact", {
			agentId: id,
			path: candidatePath,
			error: error instanceof Error ? error.message : String(error),
		});
	}
	const sidecarPath = path.join(artifactsDir, `${id}.json`);
	if (outputPath && structuredOutput && Object.hasOwn(structuredOutput, "data")) {
		try {
			const serialized = JSON.stringify(structuredOutput.data, null, 2);
			// `structured.data === undefined` (an own "data" key holding
			// `undefined`) makes JSON.stringify return `undefined` too —
			// neither a write nor a removal, which would leave a stale
			// sidecar from an earlier turn answering agent://<id>/<field>
			// with superseded data (PR #10625 review).
			if (serialized !== undefined) await writeArtifact(sidecarPath, `${serialized}\n`);
			else await fs.rm(sidecarPath, { force: true });
		} catch (error) {
			logger.warn("Failed to persist subagent structured output sidecar", {
				agentId: id,
				path: sidecarPath,
				error: error instanceof Error ? error.message : String(error),
			});
			// A stale sidecar from an earlier turn must not outlive this
			// turn's <id>.md just because the replacement write failed —
			// agent://<id>/<field> would keep answering with superseded data.
			try {
				await fs.rm(sidecarPath, { force: true });
			} catch (rmError) {
				logger.warn("Failed to drop stale subagent structured output sidecar", {
					agentId: id,
					path: sidecarPath,
					error: rmError instanceof Error ? rmError.message : String(rmError),
				});
			}
		}
	} else if (outputPath) {
		// This turn republished <id>.md; a stale sidecar from an earlier
		// turn would keep answering agent://<id>/<field> with the
		// superseded payload, so it must not outlive its <id>.md.
		try {
			await fs.rm(sidecarPath, { force: true });
		} catch (error) {
			logger.warn("Failed to drop stale subagent structured output sidecar", {
				agentId: id,
				path: sidecarPath,
				error: error instanceof Error ? error.message : String(error),
			});
		}
	}
	return { outputPath, outputMeta };
}
