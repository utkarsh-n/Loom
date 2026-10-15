import SwiftUI
import OfflineIntelligence

struct OfflineAssistantView: View {
  @Bindable var model: LoomModel
  @Environment(\.dismiss) private var dismiss
  @AppStorage(OfflineModelChoice.preferenceKey) private var modelChoiceRaw = OfflineModelChoice.defaultChoice.rawValue

  private var assistant: LoomOfflineAssistant { model.offlineAssistant }

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(alignment: .leading, spacing: 20) {
          VStack(alignment: .leading, spacing: 8) {
            Label("Offline explanations", systemImage: "cpu")
              .font(.title3.weight(.semibold))
              .foregroundStyle(Theme.cyan)
            Text("Ask about the current instrument. Its measured values are supplied as context; the answer is generated text and should be checked against the instrument.")
              .font(.callout)
              .foregroundStyle(.secondary)
          }
          VStack(alignment: .leading, spacing: 10) {
            Picker("Explanation model", selection: $modelChoiceRaw) {
              ForEach(OfflineModelChoice.allCases) { choice in
                Text(choice.title).tag(choice.rawValue)
              }
            }
            .pickerStyle(.segmented)
            .accessibilityIdentifier("offlineExplanationModel")
            Text(assistant.choice.detail)
              .font(.caption)
              .foregroundStyle(.secondary)
            Text("The pipeline keeps its selected instrument model. Its GPU weights are released while this explanation model is in use.")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
          VStack(alignment: .leading, spacing: 10) {
            Text("QUESTION").eyebrow()
            TextEditor(text: Binding(get: { assistant.question }, set: { assistant.question = $0 }))
              .frame(minHeight: 88)
              .scrollContentBackground(.hidden)
              .padding(10)
              .background(Theme.well, in: .rect(cornerRadius: 14))
              .accessibilityLabel("Question about the instrument")
            Button {
              assistant.isBusy ? assistant.cancel() : assistant.answer()
            } label: {
              Label(assistant.isBusy ? "Stop explanation" : "Ask offline", systemImage: assistant.isBusy ? "stop.fill" : "sparkle")
                .font(.headline)
                .frame(maxWidth: .infinity, minHeight: 44)
            }
            .buttonStyle(.glassProminent)
            .tint(Theme.cyan.opacity(0.75))
            .disabled(!assistant.isBusy && !assistant.isPrepared)
            .accessibilityIdentifier("askOffline")
          }
          HStack(alignment: .top, spacing: 10) {
            if assistant.isBusy { ProgressView().controlSize(.small) }
            VStack(alignment: .leading, spacing: 6) {
              Text(assistant.status).font(.caption.monospaced())
              if let error = assistant.error { Text(error).font(.caption).foregroundStyle(Theme.orange) }
            }
          }
          if !assistant.output.isEmpty {
            Text(assistant.output)
              .font(.system(.body, design: .serif))
              .textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading)
              .padding(16)
              .background(Theme.panel, in: .rect(cornerRadius: 18))
          }
          if let metrics = assistant.metrics {
            VStack(alignment: .leading, spacing: 4) {
              Text("\(metrics.generatedTokens) generated tokens · \(String(format: "%.1f", metrics.tokensPerSecond)) decode tok/s")
              if let first = metrics.firstTokenSeconds {
                Text("First token including prompt processing: \(String(format: "%.2f", first)) s")
              }
            }
            .font(.caption.monospaced())
            .foregroundStyle(Theme.muted)
          }
        }
        .padding(20)
      }
      .background(Theme.background)
      .navigationTitle("Ask offline")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
      .task { await model.prepareOfflineAssistant() }
      .onChange(of: modelChoiceRaw) { _, raw in
        Task { await assistant.select(OfflineModelChoice(rawValue: raw) ?? .defaultChoice) }
      }
    }
  }
}
