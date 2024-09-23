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
#include "../extern/turbulence_generator/TurbGen.h"

namespace quokka::TurbulentDriving{

    template <typename problem_t> auto computeForceField(const amrex::Box& box, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes, TurbGen &tg){

        amrex::FArrayBox fab(box, AMREX_SPACEDIM);

        amrex::Array4<amrex::Real> Field = fab.array();

        tg.get_turb_vector_unigrid(fab, cellSizes);
        return Field;
    }

 template <typename problem_t> auto computeDriving(amrex::MultiFab &mf, const amrex::Real dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes, TurbGen &tg) -> bool
{
	const Real dt = dt_in;

	const auto &ba = mf.boxArray();
	const auto &dmap = mf.DistributionMap();
	amrex::iMultiFab nsubstepsMF(ba, dmap, 1, 0);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);

        auto const &forceField = computeForceField<problem_t>(indexRange, cellSizes, tg);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);

            amrex::Real dE =0;

            for (int m =0; m<AMREX_SPACEDIM;m++){
                const amrex::Real dMom = forceField(i,j,k,m) * dt;

                state(i, j, k, HydroSystem<problem_t>::x1Momentum_index + m) += dMom;
                dE += dMom * dMom / (2 * rho);
            }

			state(i, j, k, HydroSystem<problem_t>::energy_index) += dE;
		});
	}
	return true;
}
}

#endif //TURBULENTDRIVING_HPP
