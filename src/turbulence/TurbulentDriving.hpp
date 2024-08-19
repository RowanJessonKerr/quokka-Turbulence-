#ifndef TURBULENTDRIVING_HPP
#define TURBULENTDRIVING_HPP

#include "AMReX.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_iMultiFab.H"

#include "fmt/core.h"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/FastMath.hpp"
#include "radiation/radiation_system.hpp"

namespace quokka::TurbulentDriving{
template <typename problem_t> auto computeDriving(amrex::MultiFab &mf, const amrex::Real dt_in, const amrex::Real levelVolume) -> bool
{
	const Real dt = dt_in;

	const auto &ba = mf.boxArray();
	const auto &dmap = mf.DistributionMap();
	amrex::iMultiFab nsubstepsMF(ba, dmap, 1, 0);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const amrex::Real x1Mom = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const amrex::Real x2Mom = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const amrex::Real x3Mom = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const amrex::Real Egas = state(i, j, k, HydroSystem<problem_t>::energy_index);

			const amrex::Real dXMom = 0.1;
			const amrex::Real dE = dXMom * dXMom / 2 * (rho * levelVolume);

			state(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = x1Mom + dXMom;
			state(i, j, k, HydroSystem<problem_t>::energy_index) = Egas;
		});
	}
	return true;
}
}

#endif //TURBULENTDRIVING_HPP
