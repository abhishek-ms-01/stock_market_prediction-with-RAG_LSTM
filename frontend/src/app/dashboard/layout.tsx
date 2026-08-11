import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen bg-background overflow-hidden text-foreground">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <Header />
        <main className="flex-1 overflow-y-auto p-8 relative">
          {/* Cinematic Lighting Elements */}
          <div className="fixed top-[-10%] left-[-5%] w-[40%] h-[40%] rounded-full bg-primary/10 blur-[120px] pointer-events-none"></div>
          <div className="fixed bottom-[-10%] right-[-5%] w-[40%] h-[40%] rounded-full bg-blue-600/10 blur-[120px] pointer-events-none"></div>
          <div className="absolute inset-0 bg-[url('/bg-grid.svg')] bg-repeat opacity-[0.15] pointer-events-none"></div>
          
          <div className="max-w-7xl mx-auto space-y-6 relative z-10">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
