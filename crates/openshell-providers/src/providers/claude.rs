// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    ProviderDiscoverySpec, ProviderError, ProviderPlugin, RealDiscoveryContext, discover_with_spec,
};

pub struct ClaudeProvider;

/// Well-known key used to transport GCP service-account JSON content through
/// the credential pipeline. The sandbox materialises this to a file on disk.
pub const GCP_CREDENTIALS_DATA_KEY: &str = "GOOGLE_APPLICATION_CREDENTIALS_DATA";

pub const SPEC: ProviderDiscoverySpec = ProviderDiscoverySpec {
    id: "claude",
    credential_env_vars: &["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    config_env_vars: &[
        "CLAUDE_CODE_USE_VERTEX",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "ANTHROPIC_VERTEX_REGION",
        "CLOUD_ML_PROJECT_ID",
        "CLOUD_ML_REGION",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
    ],
};

impl ProviderPlugin for ClaudeProvider {
    fn id(&self) -> &'static str {
        SPEC.id
    }

    fn discover_existing(&self) -> Result<Option<crate::DiscoveredProvider>, ProviderError> {
        let mut discovered = discover_with_spec(&SPEC, &RealDiscoveryContext)?;

        // GCP credentials are file-based (service-account key or ADC JSON).
        // The file won't exist inside the sandbox container, so we read the
        // contents here and transport them as a credential.  The sandbox will
        // materialise the content back to a file at a well-known path.
        if let Some(ref mut provider) = discovered {
            if let Some(contents) = read_gcp_credentials() {
                provider
                    .credentials
                    .insert(GCP_CREDENTIALS_DATA_KEY.to_string(), contents);
            }
        }

        Ok(discovered)
    }

    fn credential_env_vars(&self) -> &'static [&'static str] {
        SPEC.credential_env_vars
    }
}

/// Try to read GCP credentials JSON from the environment.
///
/// Checks in order:
/// 1. `GOOGLE_APPLICATION_CREDENTIALS` env var (explicit service-account key path)
/// 2. Default Application Default Credentials (ADC) path:
///    `~/.config/gcloud/application_default_credentials.json`
fn read_gcp_credentials() -> Option<String> {
    // 1. Explicit env var
    if let Some(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        match std::fs::read_to_string(&path) {
            Ok(contents) => return Some(contents),
            Err(e) => {
                tracing::warn!(
                    path = %path,
                    error = %e,
                    "Could not read GOOGLE_APPLICATION_CREDENTIALS file"
                );
            }
        }
    }

    // 2. Default ADC path (written by `gcloud auth application-default login`)
    if let Some(home) = std::env::var("HOME")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        let adc_path = std::path::PathBuf::from(home)
            .join(".config/gcloud/application_default_credentials.json");
        match std::fs::read_to_string(&adc_path) {
            Ok(contents) => return Some(contents),
            Err(_) => {
                tracing::debug!(
                    path = %adc_path.display(),
                    "No ADC credentials file found"
                );
            }
        }
    }

    tracing::warn!(
        "No GCP credentials found; Vertex AI auth may not work inside the sandbox. \
         Set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default login`."
    );
    None
}

#[cfg(test)]
mod tests {
    use super::SPEC;
    use crate::discover_with_spec;
    use crate::test_helpers::MockDiscoveryContext;

    #[test]
    fn discovers_claude_env_credentials() {
        let ctx = MockDiscoveryContext::new().with_env("ANTHROPIC_API_KEY", "test-key");
        let discovered = discover_with_spec(&SPEC, &ctx)
            .expect("discovery")
            .expect("provider");
        assert_eq!(
            discovered.credentials.get("ANTHROPIC_API_KEY"),
            Some(&"test-key".to_string())
        );
    }

    #[test]
    fn discovers_vertex_ai_env_vars() {
        let ctx = MockDiscoveryContext::new()
            .with_env("CLAUDE_CODE_USE_VERTEX", "1")
            .with_env("ANTHROPIC_VERTEX_PROJECT_ID", "my-project")
            .with_env("ANTHROPIC_VERTEX_REGION", "us-east5")
            .with_env("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            .with_env("ANTHROPIC_SMALL_FAST_MODEL", "claude-haiku-4-20250514");
        let discovered = discover_with_spec(&SPEC, &ctx)
            .expect("discovery")
            .expect("provider");
        assert_eq!(
            discovered.config.get("CLAUDE_CODE_USE_VERTEX"),
            Some(&"1".to_string())
        );
        assert_eq!(
            discovered.config.get("ANTHROPIC_VERTEX_PROJECT_ID"),
            Some(&"my-project".to_string())
        );
        assert_eq!(
            discovered.config.get("ANTHROPIC_VERTEX_REGION"),
            Some(&"us-east5".to_string())
        );
        assert_eq!(
            discovered.config.get("ANTHROPIC_MODEL"),
            Some(&"claude-sonnet-4-20250514".to_string())
        );
        assert_eq!(
            discovered.config.get("ANTHROPIC_SMALL_FAST_MODEL"),
            Some(&"claude-haiku-4-20250514".to_string())
        );
        assert!(discovered.credentials.get("ANTHROPIC_API_KEY").is_none());
        // GOOGLE_APPLICATION_CREDENTIALS is handled specially in discover_existing
        // (file contents are read and stored as GOOGLE_APPLICATION_CREDENTIALS_DATA)
        // so it does not appear in config_env_vars.
    }

    #[test]
    fn discovers_vertex_ai_with_cloud_ml_env_vars() {
        let ctx = MockDiscoveryContext::new()
            .with_env("CLAUDE_CODE_USE_VERTEX", "1")
            .with_env("CLOUD_ML_PROJECT_ID", "my-project")
            .with_env("CLOUD_ML_REGION", "europe-west1");
        let discovered = discover_with_spec(&SPEC, &ctx)
            .expect("discovery")
            .expect("provider");
        assert_eq!(
            discovered.config.get("CLOUD_ML_PROJECT_ID"),
            Some(&"my-project".to_string())
        );
        assert_eq!(
            discovered.config.get("CLOUD_ML_REGION"),
            Some(&"europe-west1".to_string())
        );
    }
}
