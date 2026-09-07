# Platform Engineering Deep Dive: Tuning Kubernetes Probes for LLM Cold Starts

A common failure mode when deploying Large Language Models on Kubernetes is **Probe-Induced Crash Loops**. Unlike typical microservices that boot in milliseconds, quantized LLMs must read gigabytes of weight files and initialize memory tensor graphs before answering HTTP health requests.

---

## 1. The Naive Probe Trap (The Death Spiral)

Consider the default probe configuration frequently copied from web microservices:

```yaml
# THE FLAWED PATTERN
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 2
  periodSeconds: 3
  failureThreshold: 2
```

### The Timeline of Failure
- **T+0s:** Pod scheduled, container process starts `llama-server`.
- **T+2s:** Kubelet sends first readiness probe. `llama-server` is actively mmapping 2.12 GB of weights from disk; HTTP port is not yet bound. Probe 1 fails (Connection refused).
- **T+5s:** Kubelet sends second readiness probe. Model is still loading (at 45%). Probe 2 fails.
- **T+8s:** Kubelet reaches `failureThreshold: 2`. The pod is deemed unready.
- **If applied to a `livenessProbe`:** Kubelet issues `SIGKILL` and restarts the container. The model never reaches a ready state, entering `CrashLoopBackOff` despite having zero bugs in the application code!

---

## 2. Probe Mechanics for Localized LLMs

When serving GGUF models on CPU nodes:
1. **Mmap / File Read Latency:** Reading a 2.1 GB file on standard cloud block storage (e.g. AWS gp3, GCP pd-balanced, or local SATA SSD) takes between **15 and 25 seconds**.
2. **Context Memory Allocation:** Allocating scratch space for attention heads and KV cache takes another **2 to 4 seconds**.
3. **Total Boot Time:** Real-world cold starts range from **18 to 30 seconds**.

---

## 3. Best Practice: StartupProbe + ReadinessProbe Separation

For modern Kubernetes clusters (v1.20+), the recommended pattern is using a `startupProbe` to shield the container during weight loading, paired with a lightweight `readinessProbe` and `livenessProbe`:

```yaml
# RECOMMENDED PRODUCTION PATTERN
startupProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 2
  failureThreshold: 12  # Allows up to 10s + (5s * 12) = 70 seconds for cold start

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  periodSeconds: 5
  timeoutSeconds: 2
  failureThreshold: 2   # Quickly removes pod from Service endpoints if overloaded

livenessProbe:
  httpGet:
    path: /health
    port: 8080
  periodSeconds: 15
  timeoutSeconds: 3
  failureThreshold: 3
```

### Backward-Compatible Tuning (Used in `03-deployment-fixed.yaml`)
If avoiding `startupProbe` complexity, size the `readinessProbe` conservatively:
- `initialDelaySeconds: 20`
- `periodSeconds: 5`
- `timeoutSeconds: 3`
- `failureThreshold: 10` (permits up to 70s total startup window before failing)

This ensures the pod is never marked healthy before the weights are fully loaded into RAM, while preventing premature restarts.
