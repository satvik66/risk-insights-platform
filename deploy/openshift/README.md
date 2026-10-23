# OpenShift deployment notes

The generic Kubernetes API manifest is a starting point. Build/push an OpenShift-compatible non-root image, grant writable paths where needed, inject secrets via OpenShift Secrets, apply restricted SCC policies, and configure an approved image registry. For Spark jobs, install and configure a supported Spark Operator or approved Spark execution platform. **These templates have not been deployed to a live cluster.**

Promote images by digest between environments; verify probes, health, metrics and model version during a canary. Model artifacts belong in an authenticated model registry or object-store volume, not inside application source code for production use.
