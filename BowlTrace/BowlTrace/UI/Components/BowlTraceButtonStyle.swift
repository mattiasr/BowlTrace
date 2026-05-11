import SwiftUI

// MARK: - Primary button (filled orange pill)
struct PrimaryButtonStyle: ButtonStyle {
    var isLoading = false

    func makeBody(configuration: Configuration) -> some View {
        ZStack {
            Capsule()
                .fill(Color.btAccent)
            if isLoading {
                ProgressView()
                    .tint(.white)
                    .transition(.opacity)
            } else {
                configuration.label
                    .font(.system(size: 17, weight: .semibold, design: .default))
                    .foregroundColor(.white)
                    .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 56)
        .opacity(configuration.isPressed ? 0.8 : 1.0)
        .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
        .animation(.spring(response: 0.25, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

// MARK: - Secondary button (outlined pill)
struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        ZStack {
            Capsule()
                .stroke(Color.btAccent, lineWidth: 1.5)
            configuration.label
                .font(.system(size: 17, weight: .medium))
                .foregroundColor(.btAccent)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 56)
        .opacity(configuration.isPressed ? 0.7 : 1.0)
        .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
        .animation(.spring(response: 0.25, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

// MARK: - Ghost button (text only)
struct GhostButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .medium))
            .foregroundColor(.btTextSecondary)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .opacity(configuration.isPressed ? 0.5 : 1.0)
    }
}

// MARK: - Icon button
struct IconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(.btTextPrimary)
            .frame(width: 44, height: 44)
            .background(Color.btSurfaceElevated.opacity(0.8), in: Circle())
            .opacity(configuration.isPressed ? 0.6 : 1.0)
            .scaleEffect(configuration.isPressed ? 0.93 : 1.0)
            .animation(.spring(response: 0.2, dampingFraction: 0.65), value: configuration.isPressed)
    }
}

// MARK: - View modifier helpers
extension View {
    func primaryButton(isLoading: Bool = false) -> some View {
        buttonStyle(PrimaryButtonStyle(isLoading: isLoading))
    }

    func secondaryButton() -> some View {
        buttonStyle(SecondaryButtonStyle())
    }

    func ghostButton() -> some View {
        buttonStyle(GhostButtonStyle())
    }

    func iconButton() -> some View {
        buttonStyle(IconButtonStyle())
    }
}
