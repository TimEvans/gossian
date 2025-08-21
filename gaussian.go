package gossian

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

type Gaussian struct {
	Name string
	Mean *mat.Dense
	Cov  *mat.SymDense
}

func (mg *Gaussian) Dims() int {
	rows, _ := mg.Mean.Dims()
	return rows
}

func StandardNormal(n int) *Gaussian {
	return &Gaussian{"StandardNormal", mat.NewDense(n, 1, nil), eye(n)}
}

func JointGaussian(g1, g2 *Gaussian) *Gaussian {
	var mean mat.Dense
	mean.Stack(g1.Mean, g2.Mean)
	cov := BlockSymDiag(g1.Cov, g2.Cov)
	return &Gaussian{g1.Name + "⊗" + g2.Name, &mean, cov}
}

func BlockSymDiag(m1, m2 *mat.SymDense) *mat.SymDense {
	r1, _ := m1.Dims()
	r2, _ := m2.Dims()

	off_block := mat.NewDense(r1, r2, nil)
	var top, bottom mat.Dense
	top.Augment(m1, off_block)
	bottom.Augment(off_block.T(), m2)

	var block_diag mat.Dense
	block_diag.Stack(&top, &bottom)

	return convertToSymDense(&block_diag)
}

func convertToSymDense(m *mat.Dense) *mat.SymDense {
	r, c := m.Dims()
	if r != c {
		panic("matrix must be square to convert to SymDense")
	}

	// Create full n×n data array
	data := make([]float64, r*r) // Changed from r*(r+1)/2

	// Copy all matrix elements (not just upper triangle)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			if i < j && math.Abs(m.At(i, j)-m.At(j, i)) > 1e-12 {
				panic("matrix is not symmetric")
			}
			data[i*r+j] = m.At(i, j) // Row-major order
		}
	}
	return mat.NewSymDense(r, data)
}

func eye(n int) *mat.SymDense {
	// Create n x n identity matrix
	id := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		id.SetSym(i, i, 1.0)
	}
	return id
}

func main() {
	fmt.Println(eye(2))
}
