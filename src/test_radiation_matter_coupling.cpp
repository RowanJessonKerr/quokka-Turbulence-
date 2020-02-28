//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_matter_coupling.cpp
/// \brief Defines a test problem for radiation-matter coupling.
///

#include "test_radiation_matter_coupling.hpp"

void testproblem_radiation_matter_coupling()
{
	// Problem parameters

	const int nx = 4;
	const double Lx = 1.0;
	const double CFL_number = 0.4;
	const double constant_dt = 1.0e-11; // s
	const double max_time = 1.0e-6;	    // s
	const int max_timesteps = 2e5;

	// Problem initialization

	RadSystem rad_system(RadSystem::Nx = nx, RadSystem::Lx = Lx,
			     RadSystem::CFL = CFL_number);

	auto nghost = rad_system.nghost();

	const double Erad = 1.0e12; // erg cm^-3
	const double Egas = 1.0e2;  // erg cm^-3
	const double rho = 1.0e-7;  // g cm^-3

	const double c_v =
	    rad_system.boltzmann_constant_cgs_ /
	    (rad_system.mean_molecular_mass_cgs_ * (rad_system.gamma_ - 1.0));

	std::cout << "Volumetric heat capacity c_v = " << rho * c_v << "\n";

	for (int i = nghost; i < nx + nghost; ++i) {
		rad_system.set_radEnergy(i) = Erad;
		rad_system.set_x1RadFlux(i) = 0.0;
		rad_system.set_gasEnergy(i) = Egas;
		rad_system.set_staticGasDensity(i) = rho;
	}

	std::vector<double> t;
	std::vector<double> Trad;
	std::vector<double> Tgas;
	std::vector<double> Egas_v;

	const auto initial_Erad = rad_system.ComputeRadEnergy();
	const auto initial_Egas = rad_system.ComputeGasEnergy();
	const auto initial_Etot = initial_Erad + initial_Egas;

	const auto initial_Trad =
	    std::pow(Erad / rad_system.radiation_constant(), 1. / 4.);
	const auto initial_Tgas = Egas / (rho * c_v);
	const auto kappa = rad_system.ComputeOpacity(rho, initial_Tgas);
	const auto heating_time =
	    Egas /
	    (rho * kappa * rad_system.c_light_ *
	     (rad_system.radiation_constant() * std::pow(initial_Tgas, 4) -
	      Erad));

	std::cout << "Initial radiation temperature = " << initial_Trad << "\n";
	std::cout << "Initial gas temperature = " << initial_Tgas << "\n";
	std::cout << "Heating time = " << heating_time << "\n";

	// t.push_back(rad_system.time());
	// Trad.push_back(initial_Trad);
	// Tgas.push_back(initial_Tgas);
	// Egas_v.push_back(Egas);

	// Main time loop

	for (int j = 0; j < max_timesteps; ++j) {
		if (rad_system.time() >= max_time) {
			break;
		}

		rad_system.AdvanceTimestep(constant_dt);
		rad_system.AddSourceTerms(std::make_pair(nghost, nghost + nx));

		std::cout << "Timestep " << j << "; t = " << rad_system.time()
			  << "\n";

		const auto current_Erad = rad_system.ComputeRadEnergy();
		const auto current_Egas = rad_system.ComputeGasEnergy();
		const auto current_Etot = current_Erad + current_Egas;
		const auto Ediff = std::fabs(current_Etot - initial_Etot);

		std::cout << "radiation energy = " << current_Erad << "\n";
		std::cout << "gas energy = " << current_Egas << "\n";
		std::cout << "Total energy = " << current_Etot << "\n";
		std::cout << "(Energy nonconservation = " << Ediff << ")\n";
		std::cout << "\n";

		t.push_back(rad_system.time());
		Trad.push_back(std::pow(rad_system.radEnergy(0 + nghost) /
					    rad_system.radiation_constant(),
					1. / 4.));
		Tgas.push_back(rad_system.gasEnergy(0 + nghost) / (rho * c_v));
		Egas_v.push_back(rad_system.gasEnergy(0 + nghost));
	}

	matplotlibcpp::clf();
	matplotlibcpp::yscale("log");
	matplotlibcpp::xscale("log");
	matplotlibcpp::ylim(0.1 * std::min(Tgas.front(), Trad.front()),
			    10.0 * std::max(Trad.back(), Tgas.back()));

	std::map<std::string, std::string> Trad_args;
	Trad_args["label"] = "radiation temperature";
	matplotlibcpp::plot(t, Trad, Trad_args);

	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "gas temperature";
	matplotlibcpp::plot(t, Tgas, Tgas_args);

	// matplotlibcpp::xscale("log");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4g}", rad_system.time()));
	matplotlibcpp::save(fmt::format("./radcoupling.png"));

	matplotlibcpp::clf();
	matplotlibcpp::yscale("log");
	matplotlibcpp::xscale("log");
	matplotlibcpp::ylim(0.1 * Egas_v.front(), 10.0 * Egas_v.back());

	std::map<std::string, std::string> egas_args;
	egas_args["label"] = "gas energy density";
	matplotlibcpp::plot(t, Egas_v, egas_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4g}", rad_system.time()));
	matplotlibcpp::save(fmt::format("./radcoupling_Egas.png"));

	// Cleanup and exit
	std::cout << "Finished." << std::endl;
}
