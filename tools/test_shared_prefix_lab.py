import random
import unittest

from tools.shared_prefix_lab import PageStore, capacity_estimate, run_storage_demo


class SharedPrefixLab(unittest.TestCase):
    def make_prefix(self, length=5):
        store = PageStore(page_tokens=4, record_bytes=2)
        root = store.sequence("weights-A|yarn4|int8", max_tokens=64)
        records = [bytes([i, i + 1]) for i in range(length)]
        for record in records:
            root.append(record, {"gdn": [1], "tail": [2], "ple": [3]})
        return store, root.freeze(), b"".join(records)

    def test_active_branches_keep_one_prefix_copy_after_cache_release(self):
        store, root, expected = self.make_prefix(8)
        branches = [root.fork(root.identity) for _ in range(10)]
        root.close()
        self.assertEqual(store.stats()["stored_payload_bytes"], len(expected))
        for branch in branches:
            self.assertEqual(branch.records(), expected)
        for branch in branches:
            branch.close()
            store.assert_invariants()
        self.assertEqual(store.stats()["physical_pages"], 0)

    def test_partial_page_and_running_state_are_private_on_append(self):
        store, root, expected = self.make_prefix()
        a, b = root.fork(root.identity), root.fork(root.identity)
        state = a.state
        state["gdn"][0] = 99
        state["tail"].append(100)
        a.append(b"AB", state)
        state["ple"][0] = -1  # caller mutation after append cannot mutate stored checkpoint
        self.assertEqual(a.state["ple"], [3])
        self.assertEqual(b.state["gdn"], [1])
        self.assertEqual(root.records(), expected)
        self.assertEqual(b.records(), expected)
        self.assertEqual(a.records(), expected + b"AB")
        self.assertEqual(store.stats()["copied_tail_bytes"], 2)
        for owner in (root, a, b):
            owner.close()
        store.assert_invariants()

    def test_full_page_boundary_allocates_suffix_without_copying_prefix(self):
        store, root, expected = self.make_prefix(8)
        a = root.fork(root.identity)
        a.append(b"AB", a.state)
        self.assertEqual(a.records(), expected + b"AB")
        self.assertEqual(store.stats()["copied_tail_bytes"], 0)
        root.close()
        a.close()
        store.assert_invariants()

    def test_invalid_operations_preserve_pages_and_reference_counts(self):
        store, root, expected = self.make_prefix()
        before = store.stats()
        with self.assertRaises(ValueError):
            root.fork("weights-A|yarn2|int8")
        with self.assertRaises(ValueError):
            root.append(b"AB")
        self.assertEqual(store.stats(), before)
        self.assertEqual(root.records(), expected)
        root.close()
        after = store.stats()
        root.close()  # cancellation acknowledgment may arrive more than once
        self.assertEqual(store.stats(), after)
        with self.assertRaises(ValueError):
            root.fork(root.identity)
        store.assert_invariants()

    def test_context_limit_and_bad_record_do_not_modify_shared_storage(self):
        store = PageStore(page_tokens=4, record_bytes=2)
        root = store.sequence("same", max_tokens=1)
        root.append(b"AB", {"gdn": [1]})
        branch = root.fork(root.identity)
        before = store.stats()
        for invalid in (b"CD", b"C"):
            with self.assertRaises(ValueError):
                branch.append(invalid, {"gdn": [99]})
        self.assertEqual(store.stats(), before)
        self.assertEqual(branch.state, {"gdn": [1]})
        root.close()
        branch.close()
        store.assert_invariants()

    def test_random_fork_append_close_sequences_match_flat_reference(self):
        rng = random.Random(704)
        store = PageStore(page_tokens=4, record_bytes=2)
        active = [(store.sequence("same", max_tokens=64), b"")]
        for step in range(300):
            index = rng.randrange(len(active))
            owner, expected = active[index]
            action = rng.choice(["append", "fork", "close"])
            if action == "fork" and len(active) < 12:
                active.append((owner.fork(owner.identity), expected))
            elif action == "close" and len(active) > 1:
                owner.close()
                active.pop(index)
            elif owner.length < owner.max_tokens:
                record = rng.randbytes(2)
                owner.append(record, {"step": [step]})
                active[index] = (owner, expected + record)
            store.assert_invariants()
            for branch, reference in active:
                self.assertEqual(branch.records(), reference)
        for owner, _ in active:
            owner.close()
        store.assert_invariants()
        self.assertEqual(store.stats()["physical_pages"], 0)

    def test_ten_parallel_branches_match_reference_and_release_every_page(self):
        result = run_storage_demo(10)
        self.assertTrue(result["all_branch_bytes_equal_independent_reference"])
        self.assertTrue(result["private_running_states_verified"])
        self.assertEqual(result["pages_after_all_branches_closed"], 0)
        self.assertEqual(result["shared_payload_bytes_after_append"], 4736)

    def test_capacity_distinguishes_estimate_from_actual_storage_test(self):
        result = capacity_estimate(700_000, 10_000, 10)
        self.assertEqual(result["duplicated_payload_bytes"], 97_468_800_000)
        self.assertEqual(result["ideal_shared_payload_bytes"], 10_982_400_000)
        self.assertGreaterEqual(result["shared_page_capacity_bytes"], result["ideal_shared_payload_bytes"])
        for bad in ((-1, 1, 1), (1, -1, 1), (1, 1, 0)):
            with self.assertRaises(ValueError):
                capacity_estimate(*bad)


if __name__ == "__main__":
    unittest.main()
