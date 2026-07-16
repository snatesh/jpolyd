from __future__ import annotations

from jhps import run_two_leaf_test


def main() -> None:
  tol = 5e-10
  results = []

  for D in range(1, 6):
    res = run_two_leaf_test(
      D,
      kf=2,
      vol_dim=D + 4,
      seed=4242 + D,
      verbose=True,
    )
    results.append(res)

    assert res.root_robin_residual_inf < tol, res
    assert res.interface_flux_residual_inf < tol, res
    assert res.parent_consistency_residual_inf < tol, res
    assert res.monolithic_trace_residual_inf < tol, res
    assert res.root_nb == 2 * D * 2, res  # two exterior face sets, D faces each, kf=2
    assert res.interface_nb == 2, res     # one shared face, kf=2

  print("\nall D=1..5 two-leaf dummy HPS tests passed")
  for res in results:
    print(res)


if __name__ == "__main__":
  main()
