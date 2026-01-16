import { NavLink } from "react-router-dom";

function Layout({ children }) {
  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      {/* Sidebar */}
      <aside
        style={{
          width: "240px",
          background: "#020617",
          padding: "20px",
          borderRight: "1px solid #1f2937",
          color: "#e5e7eb",
        }}
      >
        <h2 style={{ marginBottom: "30px", color: "#6366f1" }}>
          VibroSense AI
        </h2>

        <nav style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <NavLink
            to="/"
            style={({ isActive }) => ({
              color: isActive ? "#22c55e" : "#e5e7eb",
              textDecoration: "none",
              fontWeight: isActive ? 600 : 400,
            })}
          >
            Dashboard
          </NavLink>

          <NavLink
            to="/devices"
            style={({ isActive }) => ({
              color: isActive ? "#22c55e" : "#e5e7eb",
              textDecoration: "none",
              fontWeight: isActive ? 600 : 400,
            })}
          >
            Devices
          </NavLink>

          <div style={{ color: "#6b7280" }}>Models</div>
        </nav>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, padding: "24px", background: "#020617" }}>
        {children}
      </main>
    </div>
  );
}

export default Layout;
