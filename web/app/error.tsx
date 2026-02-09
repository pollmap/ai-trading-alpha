"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-center space-y-4 max-w-md">
        <div className="text-4xl">⚠️</div>
        <h2 className="text-xl font-bold text-white">Something went wrong</h2>
        <p className="text-atlas-muted text-sm">{error.message}</p>
        <button
          onClick={reset}
          className="px-4 py-2 bg-atlas-accent text-white rounded-lg text-sm hover:bg-atlas-accent/80 transition-colors"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
