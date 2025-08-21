package gossian

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestEye(t *testing.T) {
	id := eye(2)
	expected := mat.NewSymDense(2, []float64{1, 0, 0, 1})
	if !mat.EqualApprox(id, expected, 1e-12) {
		t.Errorf("eye(2) = %v, want %v", id, expected)
	}
}

func TestStandardNormal(t *testing.T) {
	g := StandardNormal(2)
	expected := eye(2)
	if !mat.EqualApprox(g.Cov, expected, 1e-12) {
		t.Errorf("StandardNormal = %v, want %v", g.Cov, expected)
	}
}

func TestJointGaussian(t *testing.T) {
	g1 := StandardNormal(2)
	g2 := StandardNormal(3)
	fmt.Printf("g1 mean dims: %v\n", g1.Dims())
	fmt.Printf("g2 mean dims: %v\n", g2.Dims())
	r, c := g1.Cov.Dims()
	fmt.Printf("g1 cov dims: %v %v\n", r, c)
	g3 := JointGaussian(g1, g2)
	expected := eye(5)
	if !mat.EqualApprox(g3.Cov, expected, 1e-12) {
		t.Errorf("JointGaussian = %v, want %v", g3.Cov, expected)
	}
}
