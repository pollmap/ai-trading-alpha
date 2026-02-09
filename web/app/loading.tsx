export default function Loading() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-center space-y-4">
        <div className="w-12 h-12 border-4 border-atlas-border border-t-atlas-accent rounded-full animate-spin mx-auto" />
        <p className="text-atlas-muted text-sm">Loading...</p>
      </div>
    </div>
  );
}
