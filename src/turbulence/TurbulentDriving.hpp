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

    template <typename problem_t> auto computeForceField(const amrex::Box box, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes){

        amrex::Dim3 lowVec = (amrex::Dim3 &&) box.smallEnd();
        amrex::Dim3 highVec = (amrex::Dim3 &&) box.bigEnd();
        const auto &length = box.length3d();

        auto* fieldBaseArray = new amrex::Real[length[0] * length[1] * length[2]* AMREX_SPACEDIM];

        amrex::Array4<amrex::Real> Field(fieldBaseArray,lowVec, highVec,AMREX_SPACEDIM);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Field(i,j,k,0) = sin(i * cellSizes[0]);
            Field(i,j,k,1) = sin(j * cellSizes[1]);
        });

        return Field;
    }

template <typename problem_t> auto computeDriving(amrex::MultiFab &mf, const amrex::Real dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes) -> bool
{
	const Real dt = dt_in;

	const auto &ba = mf.boxArray();
	const auto &dmap = mf.DistributionMap();
	amrex::iMultiFab nsubstepsMF(ba, dmap, 1, 0);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);

        auto const &forceField = computeForceField<problem_t>(indexRange, cellSizes);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);

			const auto TargetForce  = amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>(
                    AMREX_D_DECL(forceField(i,j,k,0),forceField(i,j,k,1),forceField(i,j,k,2)));

            amrex::Real dE =0;

            for (int m =0; m<AMREX_SPACEDIM;m++){
                const amrex::Real dMom = TargetForce[i] * dt;
                state(i, j, k, HydroSystem<problem_t>::x1Momentum_index + i) += dMom;
                dE += dMom * dMom / (2 * rho);
            }

			state(i, j, k, HydroSystem<problem_t>::energy_index) += dE;
		});
	}
	return true;
}
}

#endif //TURBULENTDRIVING_HPP
