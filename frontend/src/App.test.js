import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the upload screen with samples', () => {
  render(<App />);
  expect(screen.getByRole('heading', { name: /Jewellery Analyzer/i })).toBeInTheDocument();
  expect(screen.getByText(/Identify Item/i)).toBeInTheDocument();
  expect(screen.getAllByRole('img').length).toBeGreaterThanOrEqual(6);
});
