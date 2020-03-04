/*
 This file is part of MPFR C++, a C++ interface to MPFR library.
 Project homepage: http://www.holoborodko.com/pavel/
 Contact e-mail:   pavel@holoborodko.com

 Copyright (C) 2010 Pavel Holoborodko <pavel@holoborodko.com>
 Copyright (C) 2010 Konstantin Holoborodko <konstantin@holoborodko.com>

 MPFR C++ is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 MPFR C++ is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOS
 E.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

 Contributors:
 Brian Gladman, Helmut Jarausch, Fokko Beekhof, Ulrich Mutze,
 Heinz van Saanen, Pere Constans, Dmitriy Gubanov
*/

#ifndef MPREALSUPPORT_H
#define MPREALSUPPORT_H

#include "mpreal.h"
#include <ctime>
#include "Eigen/Core"

namespace Eigen {

	template<> struct NumTraits<mpfr::mpreal>
	{
		typedef mpfr::mpreal Real;
		typedef mpfr::mpreal FloatingPoint;
		enum {
			IsComplex = 0,
			HasFloatingPoint = 1,
			ReadCost = 1,
			AddCost = 1,
			MulCost = 1
		};
	};

	inline template<typename a> a machine_epsilon()
	{
		return mpfr::machine_epsilon(mpfr::mpreal::get_default_prec());
	}

	inline template<> mpfr::mpreal precision<mpfr::mpreal>()
	{
		return machine_epsilon<mpfr::mpreal>();
	}

	inline template<> mpfr::mpreal ei_random<mpfr::mpreal>()
	{
#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,0,0))
		static gmp_randstate_t state;
		static bool isFirstTime = true;

		if(isFirstTime)
		{
			gmp_randinit_default(state);
			gmp_randseed_ui(state,(unsigned)time(NULL));
			isFirstTime = false;
		}

		return mpfr::urandom(state);
#else
		return (mpfr::mpreal)((double)std::rand()/(double)RAND_MAX);
#endif
	}

	inline template<> mpfr::mpreal ei_random<mpfr::mpreal>(const mpfr::mpreal a, const mpfr::mpreal b)
	{
		return a + (b-a) * ei_random<mpfr::mpreal>();
	}
}

namespace mpfr {

  inline const mpreal& ei_conj(const mpreal& x)  { return x; }
  inline const mpreal& ei_real(const mpreal& x)  { return x; }
  inline mpreal ei_imag(const mpreal&)    { return 0.0; }
  inline mpreal ei_abs(const mpreal&  x)  { return fabs(x); }
  inline mpreal ei_abs2(const mpreal& x)  { return x*x; }
  inline mpreal ei_sqrt(const mpreal& x)  { return sqrt(x); }
  inline mpreal ei_exp(const mpreal&  x)  { return exp(x); }
  inline mpreal ei_log(const mpreal&  x)  { return log(x); }
  inline mpreal ei_sin(const mpreal&  x)  { return sin(x); }
  inline mpreal ei_cos(const mpreal&  x)  { return cos(x); }
  inline mpreal ei_pow(const mpreal& x, mpreal& y)  { return pow(x, y); }

  inline bool ei_isMuchSmallerThan(const mpreal& a, const mpreal& b, const mpreal& prec)
  {
	  return ei_abs(a) <= abs(b) * prec;
  }

  inline bool ei_isApprox(const mpreal& a, const mpreal& b, const mpreal& prec)
  {
	return ei_abs(a - b) <= min(abs(a), abs(b)) * prec;
  }

  inline bool ei_isApproxOrLessThan(const mpreal& a, const mpreal& b, const mpreal& prec)
  {
    return a <= b || ei_isApprox(a, b, prec);
  }
}

#endif // MPREALSUPPORT_H
