import LoomEngine
import SwiftUI
import UIKit

@main
struct LoomApp: App {
  @State private var model = LoomModel()
  @Environment(\.scenePhase) private var scenePhase

  var body: some Scene {
    WindowGroup {
      RootView(model: model)
        .preferredColorScheme(.dark)
        .tint(Theme.phosphor)
        .task { await model.load() }
        .onChange(of: scenePhase) { _, phase in
          switch phase {
          case .active:
            model.haptics.start()
            model.resume()
          case .background:
            model.suspend()
            model.haptics.stop()
          default: break
          }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
          model.receivedMemoryWarning()
        }
    }
  }
}

struct RootView: View {
  @Bindable var model: LoomModel
  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  var body: some View {
    Group {
      switch model.phase {
      case .loading:
        ProgressView("Loading the model…")
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      case .failed(let why):
        ContentUnavailableView {
          Label("Couldn't load the model", systemImage: "exclamationmark.triangle")
        } description: {
          Text(why)
        } actions: {
          Button("Try again") { Task { await model.load() } }.buttonStyle(.glassProminent)
        }
      case .ready:
        tabs
      }
    }
    .background(Theme.background.ignoresSafeArea())
    .sheet(item: $model.sheet, onDismiss: { model.offlineAssistantDidClose() }) { sheet in
      switch sheet {
      case .fieldNotes: FieldNotesView(model: model)
      case .settings: SettingsView(model: model)
      case .offlineAssistant: OfflineAssistantView(model: model)
      }
    }
    .sheet(item: $model.inspector) { InspectorSheet(model: model, inspector: $0) }
    .fullScreenCover(isPresented: $model.sphereOpen) { SphereRide(model: model) }
    .transaction { if reduceMotion { $0.animation = nil } }
  }

  private var tabs: some View {
    TabView(selection: $model.tab) {
      Tab("Pipeline", systemImage: "flowchart", value: AppTab.loom) {
        navigation("The pipeline") { LoomTab(model: model) }
      }
      Tab("Sphere", systemImage: "circle.hexagonpath", value: AppTab.sphere) {
        navigation("The sphere") { SphereTab(model: model) }
      }
      Tab("Embeddings", systemImage: "point.3.connected.trianglepath.dotted", value: AppTab.galaxy) {
        navigation("Embedding space") { GalaxyTab(model: model) }
      }
      Tab("Tune", systemImage: "tuningfork", value: AppTab.bench) {
        navigation("Tune the model") { BenchTab(model: model) }
      }
    }
  }

  private func navigation<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
    NavigationStack {
      content()
        .background(Theme.background.ignoresSafeArea())
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .topBarLeading) {
            Button {
              model.tab = .loom
              withAnimation(.spring(duration: 0.45, bounce: 0.2)) { model.consoleOpen = true }
              model.haptics.transient(intensity: 0.4, sharpness: 0.5)
            } label: {
              Label("Dials", systemImage: "dial.medium").labelStyle(.titleAndIcon)
            }
            .accessibilityIdentifier("dials")
          }
          ToolbarItemGroup(placement: .topBarTrailing) {
            Button("Ask offline", systemImage: "bubble.left") { model.sheet = .offlineAssistant }
              .accessibilityIdentifier("offlineAssistant")
            Button("How it works", systemImage: "info.circle") { model.sheet = .fieldNotes }
              .accessibilityIdentifier("fieldNotes")
            Button("Settings", systemImage: "slider.horizontal.3") { model.sheet = .settings }
              .accessibilityIdentifier("settings")
          }
        }
    }
  }
}
