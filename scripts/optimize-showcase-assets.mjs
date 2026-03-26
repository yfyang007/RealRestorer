import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");

const VARIANTS = {
  full: {
    suffix: ".opt.webp",
    convertArgs: ["-auto-orient", "-strip", "-resize", "1600x1600>", "-quality", "82", "-define", "webp:method=6", "-define", "webp:auto-filter=true"]
  },
  thumb: {
    suffix: ".thumb.webp",
    convertArgs: ["-auto-orient", "-strip", "-thumbnail", "320x320>", "-quality", "68", "-define", "webp:method=6"]
  },
  logo: {
    suffix: ".logo.webp",
    convertArgs: ["-auto-orient", "-strip", "-resize", "160x160>", "-quality", "86", "-define", "webp:method=6"]
  }
};

function loadWindowData(relativeFile) {
  const absoluteFile = path.join(projectRoot, relativeFile);
  const source = fs.readFileSync(absoluteFile, "utf8");
  const context = { window: {} };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: absoluteFile });
  return context.window;
}

function runCommand(command, args) {
  const result = spawnSync(command, args, { stdio: "pipe", encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(result.stderr || `Failed to run ${command}`);
  }
}

function getVariantPath(relativeSourcePath, variantName) {
  const parsed = path.parse(relativeSourcePath);
  return path.join(parsed.dir, `${parsed.name}${VARIANTS[variantName].suffix}`).replaceAll(path.sep, "/");
}

function optimizeImage(relativeSourcePath, variantName) {
  const sourcePath = path.join(projectRoot, relativeSourcePath);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`Missing source image: ${relativeSourcePath}`);
  }

  const relativeTargetPath = getVariantPath(relativeSourcePath, variantName);
  const targetPath = path.join(projectRoot, relativeTargetPath);
  const sourceStat = fs.statSync(sourcePath);

  if (fs.existsSync(targetPath)) {
    const targetStat = fs.statSync(targetPath);
    if (targetStat.mtimeMs >= sourceStat.mtimeMs) {
      return {
        sourceSize: sourceStat.size,
        targetSize: targetStat.size,
        targetPath: relativeTargetPath,
        skipped: true
      };
    }
  }

  runCommand("convert", [sourcePath, ...VARIANTS[variantName].convertArgs, targetPath]);
  const targetStat = fs.statSync(targetPath);

  return {
    sourceSize: sourceStat.size,
    targetSize: targetStat.size,
    targetPath: relativeTargetPath,
    skipped: false
  };
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = -1;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

function main() {
  const showcaseWindow = loadWindowData("showcase-data.js");
  const rankingWindow = loadWindowData("ranking-data.js");

  const jobs = [];

  for (const task of showcaseWindow.showcaseData || []) {
    for (const sample of task.samples || []) {
      if (sample.input) {
        jobs.push({ source: sample.input.replace(/^\.\//, ""), variant: "full" });
      }
      if (sample.output) {
        jobs.push({ source: sample.output.replace(/^\.\//, ""), variant: "full" });
        jobs.push({ source: sample.output.replace(/^\.\//, ""), variant: "thumb" });
      }
    }
  }

  for (const entry of rankingWindow.rankingData || []) {
    if (entry.logo) {
      jobs.push({ source: entry.logo.replace(/^\.\//, ""), variant: "logo" });
    }
  }

  const uniqueJobs = Array.from(
    new Map(jobs.map((job) => [`${job.source}::${job.variant}`, job])).values()
  );

  let totalSourceBytes = 0;
  let totalTargetBytes = 0;
  let generatedCount = 0;
  let skippedCount = 0;

  for (const job of uniqueJobs) {
    const result = optimizeImage(job.source, job.variant);
    totalSourceBytes += result.sourceSize;
    totalTargetBytes += result.targetSize;
    if (result.skipped) {
      skippedCount += 1;
    } else {
      generatedCount += 1;
      console.log(`generated ${job.variant.padEnd(5)} ${result.targetPath}`);
    }
  }

  console.log("");
  console.log(`processed: ${uniqueJobs.length}`);
  console.log(`generated: ${generatedCount}`);
  console.log(`skipped:   ${skippedCount}`);
  console.log(`source:    ${formatBytes(totalSourceBytes)}`);
  console.log(`optimized: ${formatBytes(totalTargetBytes)}`);
}

main();
