/*!
# cuda-snapshot

State snapshotting and recovery.

Agents need persistence. If they crash, reboot, or get migrated,
they should resume from a known state. This crate provides
checkpoint/restore with delta encoding and versioned history.

- Full state snapshots
- Delta encoding (store only changes)
- Versioned snapshot history
- Point-in-time recovery
- Snapshot compression hints
- Rollback to any version
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A state snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: String,
    pub agent_id: String,
    pub version: u64,
    pub data: HashMap<String, Vec<u8>>,
    pub parent_id: Option<String>,  // for delta chains
    pub deltas: Vec<Delta>,
    pub is_full: bool,
    pub created_ms: u64,
    pub data_size: usize,
    pub checksum: u64,
}

/// A change delta
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Delta {
    pub key: String,
    pub old_value: Option<Vec<u8>>,
    pub new_value: Option<Vec<u8>>,
    pub operation: DeltaOp,
    pub timestamp: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaOp { Set, Delete }

/// Snapshot policy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotPolicy {
    pub max_versions: usize,
    pub auto_snapshot_interval_ms: u64,
    pub delta_threshold: usize,    // deltas before auto-full snapshot
    pub compress_after_ms: u64,
}

impl Default for SnapshotPolicy {
    fn default() -> Self { SnapshotPolicy { max_versions: 100, auto_snapshot_interval_ms: 60_000, delta_threshold: 10, compress_after_ms: 300_000 } }
}

/// The snapshot manager
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotManager {
    pub snapshots: HashMap<String, Vec<Snapshot>>,  // agent_id → version history
    pub pending_deltas: HashMap<String, Vec<Delta>>,
    pub policy: SnapshotPolicy,
    pub next_id: u64,
    pub total_snapshots: u64,
    pub total_restores: u64,
    pub total_data_bytes: usize,
}

impl SnapshotManager {
    pub fn new() -> Self { SnapshotManager { snapshots: HashMap::new(), pending_deltas: HashMap::new(), policy: SnapshotPolicy::default(), next_id: 1, total_snapshots: 0, total_restores: 0, total_data_bytes: 0 } }

    /// Take a full snapshot
    pub fn snapshot(&mut self, agent_id: &str, state: &HashMap<String, Vec<u8>>) -> String {
        let history = self.snapshots.entry(agent_id.to_string()).or_insert_with(Vec::new);
        let version = (history.len() + 1) as u64;
        let parent = history.last().map(|s| s.id.clone());
        let data_size: usize = state.values().map(|v| v.len()).sum();
        let checksum = compute_checksum(state);

        // Flush pending deltas
        self.pending_deltas.remove(agent_id);

        let snap = Snapshot { id: format!("snap_{}", self.next_id), agent_id: agent_id.to_string(), version, data: state.clone(), parent_id: parent, deltas: vec![], is_full: true, created_ms: now(), data_size, checksum };
        self.next_id += 1;
        self.total_snapshots += 1;
        self.total_data_bytes += data_size;

        history.push(snap.clone());
        self.enforce_policy(agent_id);
        snap.id
    }

    /// Record a delta (change)
    pub fn record_delta(&mut self, agent_id: &str, key: &str, old_value: Option<&[u8]>, new_value: Option<&[u8]>) {
        let op = if new_value.is_some() { DeltaOp::Set } else { DeltaOp::Delete };
        let delta = Delta { key: key.to_string(), old_value: old_value.map(|v| v.to_vec()), new_value: new_value.map(|v| v.to_vec()), operation: op, timestamp: now() };
        self.pending_deltas.entry(agent_id.to_string()).or_insert_with(Vec::new).push(delta);
    }

    /// Take a delta snapshot (requires applying deltas to last full)
    pub fn delta_snapshot(&mut self, agent_id: &str) -> Option<String> {
        let deltas = self.pending_deltas.remove(agent_id)?;
        if deltas.is_empty() { return None; }
        let history = self.snapshots.get(agent_id)?;
        let last_full = history.iter().rev().find(|s| s.is_full)?;
        let mut data = last_full.data.clone();

        // Apply deltas
        let delta_size: usize = deltas.iter().map(|d| d.new_value.as_ref().map_or(0, |v| v.len())).sum();
        for delta in &deltas {
            match delta.operation {
                DeltaOp::Set => { if let Some(ref val) = delta.new_value { data.insert(delta.key.clone(), val.clone()); } }
                DeltaOp::Delete => { data.remove(&delta.key); }
            }
        }

        let version = (history.len() + 1) as u64;
        let data_size: usize = data.values().map(|v| v.len()).sum();
        let snap = Snapshot { id: format!("snap_{}", self.next_id), agent_id: agent_id.to_string(), version, data, parent_id: Some(last_full.id.clone()), deltas, is_full: false, created_ms: now(), data_size, checksum: 0 };
        self.next_id += 1;
        self.total_snapshots += 1;
        self.total_data_bytes += delta_size;
        let history = self.snapshots.get_mut(agent_id)?;
        history.push(snap);
        self.enforce_policy(agent_id);
        history.last().map(|s| s.id.clone())
    }

    /// Restore to latest version
    pub fn restore(&mut self, agent_id: &str) -> Option<HashMap<String, Vec<u8>>> {
        let history = self.snapshots.get(agent_id)?;
        let latest = history.last()?;
        self.total_restores += 1;
        Some(latest.data.clone())
    }

    /// Restore to a specific version
    pub fn restore_version(&mut self, agent_id: &str, version: u64) -> Option<HashMap<String, Vec<u8>>> {
        let history = self.snapshots.get(agent_id)?;
        let snap = history.iter().find(|s| s.version == version)?;
        self.total_restores += 1;
        Some(snap.data.clone())
    }

    /// Get latest state without restoring
    pub fn current_state(&self, agent_id: &str) -> Option<&HashMap<String, Vec<u8>>> {
        self.snapshots.get(agent_id)?.last().map(|s| &s.data)
    }

    /// Snapshot history for an agent
    pub fn history(&self, agent_id: &str) -> Vec<&Snapshot> {
        self.snapshots.get(agent_id).map(|h| h.iter().collect()).unwrap_or_default()
    }

    /// Should we auto-snapshot?
    pub fn should_auto_snapshot(&self, agent_id: &str) -> bool {
        let pending_count = self.pending_deltas.get(agent_id).map(|d| d.len()).unwrap_or(0);
        if pending_count >= self.policy.delta_threshold { return true; }
        if let Some(history) = self.snapshots.get(agent_id) {
            if let Some(last) = history.last() {
                return now() - last.created_ms >= self.policy.auto_snapshot_interval_ms;
            }
        }
        false
    }

    /// Remove old snapshots beyond policy
    fn enforce_policy(&mut self, agent_id: &str) {
        if let Some(history) = self.snapshots.get_mut(agent_id) {
            while history.len() > self.policy.max_versions {
                if let Some(removed) = history.remove(0) {
                    self.total_data_bytes -= removed.data_size;
                }
            }
        }
    }

    /// Summary
    pub fn summary(&self) -> String {
        let agents = self.snapshots.len();
        let versions: usize = self.snapshots.values().map(|h| h.len()).sum();
        format!("Snapshots: {} agents, {} versions, {} bytes, {} restores",
            agents, versions, self.total_data_bytes, self.total_restores)
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

fn compute_checksum(data: &HashMap<String, Vec<u8>>) -> u64 {
    let mut h: u64 = 0;
    for (k, v) in data { h = h.wrapping_mul(31).wrapping_add(k.len() as u64).wrapping_add(v.len() as u64); }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> HashMap<String, Vec<u8>> {
        let mut s = HashMap::new();
        s.insert("x".into(), b"hello".to_vec());
        s.insert("y".into(), b"world".to_vec());
        s
    }

    #[test]
    fn test_full_snapshot_and_restore() {
        let mut sm = SnapshotManager::new();
        sm.snapshot("agent1", &sample_state());
        let restored = sm.restore("agent1").unwrap();
        assert_eq!(restored.get("x").unwrap(), b"hello");
    }

    #[test]
    fn test_delta_snapshot() {
        let mut sm = SnapshotManager::new();
        sm.snapshot("agent1", &sample_state());
        sm.record_delta("agent1", "x", Some(b"hello"), Some(b"updated"));
        sm.record_delta("agent1", "z", None, Some(b"new"));
        sm.delta_snapshot("agent1").unwrap();
        let restored = sm.restore("agent1").unwrap();
        assert_eq!(restored.get("x").unwrap(), b"updated");
        assert_eq!(restored.get("z").unwrap(), b"new");
    }

    #[test]
    fn test_versioned_restore() {
        let mut sm = SnapshotManager::new();
        let mut s1 = HashMap::new();
        s1.insert("v".into(), b"1".to_vec());
        sm.snapshot("agent1", &s1);
        let mut s2 = HashMap::new();
        s2.insert("v".into(), b"2".to_vec());
        sm.snapshot("agent1", &s2);
        let v1 = sm.restore_version("agent1", 1).unwrap();
        assert_eq!(v1.get("v").unwrap(), b"1");
        let v2 = sm.restore_version("agent1", 2).unwrap();
        assert_eq!(v2.get("v").unwrap(), b"2");
    }

    #[test]
    fn test_history() {
        let mut sm = SnapshotManager::new();
        sm.snapshot("agent1", &sample_state());
        sm.snapshot("agent1", &sample_state());
        assert_eq!(sm.history("agent1").len(), 2);
    }

    #[test]
    fn test_delta_delete() {
        let mut sm = SnapshotManager::new();
        sm.snapshot("agent1", &sample_state());
        sm.record_delta("agent1", "x", Some(b"hello"), None);
        sm.delta_snapshot("agent1").unwrap();
        let restored = sm.restore("agent1").unwrap();
        assert!(!restored.contains_key("x"));
    }

    #[test]
    fn test_policy_enforcement() {
        let mut sm = SnapshotManager::new();
        sm.policy.max_versions = 3;
        for i in 0..5 {
            let mut s = HashMap::new();
            s.insert("v".into(), format!("{}", i).into_bytes());
            sm.snapshot("agent1", &s);
        }
        assert_eq!(sm.history("agent1").len(), 3);
    }

    #[test]
    fn test_auto_snapshot_trigger() {
        let mut sm = SnapshotManager::new();
        sm.policy.delta_threshold = 3;
        sm.snapshot("agent1", &sample_state());
        for i in 0..3 { sm.record_delta("agent1", &format!("k{}", i), None, Some(b"v")); }
        assert!(sm.should_auto_snapshot("agent1"));
    }

    #[test]
    fn test_current_state() {
        let mut sm = SnapshotManager::new();
        sm.snapshot("agent1", &sample_state());
        let state = sm.current_state("agent1").unwrap();
        assert_eq!(state.get("x").unwrap(), b"hello");
    }

    #[test]
    fn test_summary() {
        let sm = SnapshotManager::new();
        let s = sm.summary();
        assert!(s.contains("0 agents"));
    }
}
