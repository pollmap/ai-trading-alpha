import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-center space-y-4">
        <div className="text-6xl font-bold text-atlas-border">404</div>
        <h2 className="text-xl font-bold text-white">Page Not Found</h2>
        <p className="text-atlas-muted text-sm">The page you are looking for does not exist.</p>
        <Link
          href="/"
          className="inline-block px-4 py-2 bg-atlas-accent text-white rounded-lg text-sm hover:bg-atlas-accent/80 transition-colors"
        >
          Back to Overview
        </Link>
      </div>
    </div>
  );
}
