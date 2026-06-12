import Markdown from "@/components/Markdown";
import { buildToc } from "@/lib/content";

export default function DocPage({
  markdown,
  aside,
}: {
  markdown: string;
  aside?: React.ReactNode;
}) {
  const toc = buildToc(markdown);
  return (
    <div className="mx-auto flex max-w-6xl gap-10 px-4 py-12 sm:px-6">
      <aside className="hidden w-64 shrink-0 lg:block">
        <div className="sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto pr-2">
          {aside}
          <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            On this page
          </div>
          <nav className="space-y-1.5 border-l border-white/10 text-sm">
            {toc.map((entry) => (
              <a
                key={entry.id + entry.text}
                href={`#${entry.id}`}
                className={`block border-l-2 border-transparent text-zinc-400 transition-colors hover:border-orange-400 hover:text-orange-200 ${
                  entry.depth === 1 ? "pl-3" : "pl-6 text-[13px] text-zinc-500"
                }`}
              >
                {entry.text}
              </a>
            ))}
          </nav>
        </div>
      </aside>
      <article className="min-w-0 flex-1">
        <Markdown>{markdown}</Markdown>
      </article>
    </div>
  );
}
