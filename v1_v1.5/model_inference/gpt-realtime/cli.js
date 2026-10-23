#!/usr/bin/env node

import fs from "fs";
import minimist from "minimist";
import WebSocket from "ws";
import wav from "wav";
import "dotenv/config";

const argv = minimist(process.argv.slice(2), {
  string: ["input", "output", "model", "language"],
  alias: { i: "input", o: "output", m: "model" },
});
const inputPath = argv.input;
const outputPath = argv.output || "combined.wav";
const model = argv.model || process.env.OPENAI_REALTIME_MODEL || "gpt-realtime";
const language = argv.language || "vi";
const apiKey = process.env.OPENAI_API_KEY || process.env.OPENAI_KEY;
const sampleRate = 24000;
const timeoutMs = Number(process.env.REALTIME_RESPONSE_TIMEOUT_MS || 45000);

if (!inputPath || !apiKey) {
  console.error("Usage: cli.js --input input.wav [--output combined.wav]");
  console.error("OPENAI_API_KEY is required (OPENAI_KEY is also accepted).");
  process.exit(2);
}

function readWav(path) {
  return new Promise((resolve, reject) => {
    const reader = new wav.Reader();
    const chunks = [];
    let format;
    reader.on("format", (value) => { format = value; });
    reader.on("data", (chunk) => chunks.push(chunk));
    reader.on("error", reject);
    reader.on("end", () => resolve({ format, data: Buffer.concat(chunks) }));
    fs.createReadStream(path).on("error", reject).pipe(reader);
  });
}

function monoPcm16(format, data) {
  if (!format || ![16, 32].includes(format.bitDepth)) {
    throw new Error(`Unsupported WAV bit depth: ${format?.bitDepth}`);
  }
  const bytes = format.bitDepth / 8;
  const frames = Math.floor(data.length / (bytes * format.channels));
  const source = new Int16Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    let total = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      const offset = (frame * format.channels + channel) * bytes;
      total += format.bitDepth === 16
        ? data.readInt16LE(offset)
        : Math.round(Math.max(-1, Math.min(1, data.readFloatLE(offset))) * 32767);
    }
    source[frame] = Math.round(total / format.channels);
  }
  if (format.sampleRate === sampleRate) return source;
  const ratio = sampleRate / format.sampleRate;
  const target = new Int16Array(Math.floor(source.length * ratio));
  for (let i = 0; i < target.length; i += 1) {
    const position = i / ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, source.length - 1);
    const fraction = position - left;
    target[i] = Math.round(source[left] * (1 - fraction) + source[right] * fraction);
  }
  return target;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function writePcmWav(path, pcm) {
  return new Promise((resolve, reject) => {
    const writer = new wav.Writer({ sampleRate, channels: 1, bitDepth: 16 });
    const stream = fs.createWriteStream(path);
    stream.on("finish", resolve);
    stream.on("error", reject);
    writer.pipe(stream);
    writer.end(pcm);
  });
}

const { format, data } = await readWav(inputPath);
const input = monoPcm16(format, data);
const outputChunks = [];
const timelineEvents = [];
const responsePlayoutEnds = new Map();
const responseCutoffs = new Map();
let streamStartedAtMs = null;
let sessionReady = false;
let activeResponses = 0;
let completedResponses = 0;
let lastEventAt = Date.now();
let fatalError;

const socket = new WebSocket(`wss://api.openai.com/v1/realtime?model=${encodeURIComponent(model)}`, {
  headers: { Authorization: `Bearer ${apiKey}` },
});

socket.on("open", () => {
  socket.send(JSON.stringify({
    type: "session.update",
    session: {
      type: "realtime",
      output_modalities: ["audio"],
      instructions: `Luôn trả lời bằng tiếng Việt. Ngôn ngữ đầu vào là ${language}.`,
      audio: {
        input: {
          format: { type: "audio/pcm", rate: sampleRate },
          turn_detection: {
            type: "server_vad",
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
            create_response: true,
            interrupt_response: true,
          },
        },
        output: {
          format: { type: "audio/pcm", rate: sampleRate },
          voice: process.env.OPENAI_REALTIME_VOICE || "alloy",
        },
      },
    },
  }));
});

socket.on("message", (raw) => {
  lastEventAt = Date.now();
  const event = JSON.parse(raw.toString());
  const timelineTime = streamStartedAtMs === null
    ? null
    : Math.max(0, (performance.now() - streamStartedAtMs) / 1000);
  if (timelineTime !== null && [
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "response.created",
    "response.done",
  ].includes(event.type)) {
    timelineEvents.push({
      type: event.type,
      time: Number(timelineTime.toFixed(6)),
      response_id: event.response?.id || event.response_id || null,
      status: event.response?.status || null,
    });
  }
  if (event.type === "session.updated") sessionReady = true;
  if (event.type === "response.created") activeResponses += 1;
  if (event.type === "response.done") {
    activeResponses = Math.max(0, activeResponses - 1);
    completedResponses += 1;
    if (event.response?.status === "cancelled" && timelineTime !== null) {
      const responseId = event.response?.id || event.response_id;
      const cutoffSample = Math.round(timelineTime * sampleRate);
      if (responseId) {
        responseCutoffs.set(responseId, cutoffSample);
        responsePlayoutEnds.set(responseId, cutoffSample);
      }
    }
  }
  if (event.type === "error") {
    fatalError = new Error(JSON.stringify(event.error));
  }
  if (event.type === "response.output_audio.delta" || event.type === "response.audio.delta") {
    const chunk = Buffer.from(event.delta, "base64");
    const arrivalSample = Math.max(
      0,
      Math.round((performance.now() - streamStartedAtMs) * sampleRate / 1000),
    );
    const responseId = event.response_id || "unknown";
    const responseEnd = responsePlayoutEnds.get(responseId) || 0;
    const startSample = Math.max(arrivalSample, responseEnd);
    outputChunks.push({ responseId, startSample, chunk });
    responsePlayoutEnds.set(responseId, startSample + chunk.length / 2);
  }
});
socket.on("error", (error) => { fatalError = error; });

const connectDeadline = Date.now() + 15000;
while (!sessionReady && !fatalError && Date.now() < connectDeadline) await sleep(25);
if (fatalError) throw fatalError;
if (!sessionReady) throw new Error("Timed out waiting for session.updated");

const frameSamples = Math.floor(sampleRate / 10);
streamStartedAtMs = performance.now();
lastEventAt = Date.now();
for (let offset = 0; offset < input.length; offset += frameSamples) {
  // Pace against the absolute source timeline. This avoids accumulating
  // setTimeout drift over long clips.
  const targetTimeMs = streamStartedAtMs + (offset / sampleRate) * 1000;
  const waitMs = targetTimeMs - performance.now();
  if (waitMs > 0) await sleep(waitMs);
  const frame = input.subarray(offset, Math.min(offset + frameSamples, input.length));
  socket.send(JSON.stringify({
    type: "input_audio_buffer.append",
    audio: Buffer.from(frame.buffer, frame.byteOffset, frame.byteLength).toString("base64"),
  }));
  if (fatalError) throw fatalError;
}
const inputEndTimeMs = streamStartedAtMs + (input.length / sampleRate) * 1000;
const finalPaceWaitMs = inputEndTimeMs - performance.now();
if (finalPaceWaitMs > 0) await sleep(finalPaceWaitMs);

const responseDeadline = Date.now() + timeoutMs;
while (Date.now() < responseDeadline) {
  if (fatalError) throw fatalError;
  if (completedResponses > 0 && activeResponses === 0 && Date.now() - lastEventAt >= 1500) break;
  await sleep(50);
}
socket.close();

// Silence is a valid full-duplex action (for example, continued listening).
// Preserve the input timeline so downstream scorers still receive a valid WAV.
let outputEndSample = input.length;
for (const { responseId, startSample, chunk } of outputChunks) {
  const cutoff = responseCutoffs.get(responseId) ?? Infinity;
  const effectiveEnd = Math.min(startSample + chunk.length / 2, cutoff);
  outputEndSample = Math.max(outputEndSample, effectiveEnd);
}
const output = Buffer.alloc(outputEndSample * 2);
for (const { responseId, startSample, chunk } of outputChunks) {
  const cutoff = responseCutoffs.get(responseId) ?? Infinity;
  const effectiveSamples = Math.max(
    0,
    Math.min(chunk.length / 2, cutoff - startSample, outputEndSample - startSample),
  );
  if (effectiveSamples > 0) chunk.copy(output, startSample * 2, 0, effectiveSamples * 2);
}
const gptPath = outputPath.replace(/\.wav$/i, "_gpt_response.wav");
const timingPath = outputPath.replace(/\.wav$/i, "_timing.json");
await writePcmWav(gptPath, output);
fs.writeFileSync(timingPath, JSON.stringify({
  schema_version: 1,
  clock: "monotonic_from_first_input_frame",
  sample_rate: sampleRate,
  input_duration: input.length / sampleRate,
  output_timeline_duration: outputEndSample / sampleRate,
  events: timelineEvents,
}, null, 2));
console.log(`responses=${completedResponses} output=${gptPath} duration=${(outputEndSample / sampleRate).toFixed(3)}s`);
