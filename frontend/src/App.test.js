import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the upload screen', () => {
  render(<App />);
  expect(screen.getByText(/Jewellery Analyzer/i)).toBeInTheDocument();
  expect(screen.getByText(/Identify Item/i)).toBeInTheDocument();
});
