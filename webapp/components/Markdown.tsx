import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { slugify } from "@/lib/content";
import CodeBlock from "./CodeBlock";
import type { ReactNode } from "react";

function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (node && typeof node === "object" && "props" in node) {
    return textOf((node as { props: { children?: ReactNode } }).props.children);
  }
  return "";
}

function heading(Tag: "h1" | "h2" | "h3" | "h4", className: string) {
  return function Heading({ children }: { children?: ReactNode }) {
    const id = slugify(textOf(children));
    return (
      <Tag id={id} className={`group scroll-mt-24 ${className}`}>
        <a href={`#${id}`} className="no-underline">
          {children}
        </a>
        <a
          href={`#${id}`}
          className="ml-2 text-orange-400/0 transition-colors group-hover:text-orange-400/70"
          aria-hidden
        >
          #
        </a>
      </Tag>
    );
  };
}

export default function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown max-w-none text-[15px] leading-relaxed text-zinc-300">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: heading("h1", "mt-2 mb-4 text-3xl font-bold text-zinc-100"),
          h2: heading("h2", "mt-10 mb-3 border-b border-white/10 pb-2 text-2xl font-semibold text-zinc-100"),
          h3: heading("h3", "mt-8 mb-2 text-xl font-semibold text-zinc-100"),
          h4: heading("h4", "mt-6 mb-2 text-base font-semibold text-zinc-200"),
          p: ({ children }) => <p className="my-3">{children}</p>,
          ul: ({ children }) => <ul className="my-3 list-disc space-y-1 pl-6">{children}</ul>,
          ol: ({ children }) => <ol className="my-3 list-decimal space-y-1 pl-6">{children}</ol>,
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-orange-300 underline decoration-orange-300/40 underline-offset-2 hover:decoration-orange-300"
            >
              {children}
            </a>
          ),
          strong: ({ children }) => <strong className="font-semibold text-zinc-100">{children}</strong>,
          blockquote: ({ children }) => (
            <blockquote className="my-4 border-l-2 border-orange-400/50 pl-4 text-zinc-400">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-8 border-white/10" />,
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto rounded-xl border border-white/10">
              <table className="w-full text-left text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="border-b border-white/10 bg-white/[0.04] text-xs uppercase tracking-wide text-zinc-400">
              {children}
            </thead>
          ),
          th: ({ children }) => <th className="px-3 py-2 font-medium">{children}</th>,
          tr: ({ children }) => <tr className="border-b border-white/5 last:border-0">{children}</tr>,
          td: ({ children }) => <td className="px-3 py-2 align-top">{children}</td>,
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children }) => {
            const match = /language-(\w+)/.exec(className || "");
            const raw = textOf(children).replace(/\n$/, "");
            if (match || raw.includes("\n")) {
              return (
                <CodeBlock language={match?.[1]} code={raw}>
                  <code className={className}>{children}</code>
                </CodeBlock>
              );
            }
            return (
              <code className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[13px] text-orange-200">
                {children}
              </code>
            );
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
