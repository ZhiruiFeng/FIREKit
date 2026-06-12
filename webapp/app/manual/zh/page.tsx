import type { Metadata } from "next";
import Link from "next/link";
import { readContentFile } from "@/lib/content";
import DocPage from "@/components/DocPage";

export const metadata: Metadata = {
  title: "用户手册",
  description: "FIREKit 用户手册（中文版）：安装、核心工作流、各产品指南与端到端流水线。",
};

export default function ManualZhPage() {
  const manual = readContentFile("MANUAL.zh-CN.md");
  return (
    <DocPage
      markdown={manual}
      aside={
        <div className="mb-6">
          <Link
            href="/manual"
            className="inline-block rounded-lg border border-white/10 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:border-orange-400/40 hover:text-orange-200"
          >
            English →
          </Link>
        </div>
      }
    />
  );
}
