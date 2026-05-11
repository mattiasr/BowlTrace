import SwiftUI

struct StatPillView: View {
    let title: String
    let value: String

    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.btTextSecondary)
                .textCase(.uppercase)
                .tracking(0.5)

            Text(value)
                .font(.system(size: 20, weight: .bold, design: .rounded))
                .foregroundColor(.btTextPrimary)
                .monospacedDigit()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.btSurface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

struct StatsRowView: View {
    let stats: BallStats

    @State private var appeared = false

    var body: some View {
        HStack(spacing: 12) {
            StatPillView(title: "Max Speed", value: String(format: "%.0fmph", stats.maxSpeedMPH))
                .opacity(appeared ? 1 : 0)
                .offset(y: appeared ? 0 : 8)
                .animation(.easeOut(duration: 0.25).delay(0.0), value: appeared)

            StatPillView(title: "Entry Angle", value: String(format: "%.1f°", stats.entryAngleDegrees))
                .opacity(appeared ? 1 : 0)
                .offset(y: appeared ? 0 : 8)
                .animation(.easeOut(duration: 0.25).delay(0.06), value: appeared)

            StatPillView(title: "Side Revs", value: "\(stats.sideRevolutions)")
                .opacity(appeared ? 1 : 0)
                .offset(y: appeared ? 0 : 8)
                .animation(.easeOut(duration: 0.25).delay(0.12), value: appeared)
        }
        .padding(.horizontal, 16)
        .onAppear { appeared = true }
    }
}
